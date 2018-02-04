! THIS VERSION: GALAHAD 2.1 - 22/03/2007 AT 09:00 GMT.
   PROGRAM GALAHAD_ROOTS_test_deck
   USE GALAHAD_ROOTS_double                       ! double precision version
   IMPLICIT NONE
   INTEGER, PARAMETER :: wp = KIND( 1.0D+0 ) ! set precision
   REAL ( KIND = wp ), PARAMETER :: one = 1.0_wp

   INTEGER :: i, order, type, nroots
   REAL ( KIND = wp ) :: root, A( 0 : 4 ), ROOTS( 4 ), C( 0 : 10 )
   REAL ( KIND = wp ) :: A1( 0 : 5 ), A2( 0: 2 ), ROOTS2( 1 )
   LOGICAL :: debug = .FALSE.
   TYPE ( ROOTS_data_type ) :: data
   TYPE ( ROOTS_control_type ) :: control        
   TYPE ( ROOTS_inform_type ) :: inform

   CALL ROOTS_initialize( data, control, inform )
   control%tol = EPSILON( one ) ** 0.75
! GO TO 10

   A( 0 ) = 5.002677833377567e+05
   A( 1 ) = -2.501338916688782e+05
   A( 2 ) = -3.277148512797464e+07
   A( 3 ) = 1.530716457767176e+07

   CALL ROOTS_cubic( A( 0 ), A( 1 ), A( 2 ), A( 3 ), control%tol,              &
                     nroots, ROOTS( 1 ), ROOTS( 2), ROOTS( 3 ), .FALSE. )

   IF ( nroots /= 0 ) WRITE( 6, "( ' roots: ', 4ES12.4 )") ROOTS( : nroots )
   IF ( nroots /= 0 ) WRITE( 6, "( ' value: ', 4ES12.4 )")                     &
     ( ROOTS_polynomial_value( ROOTS( i ), A( 0 : 3 ) ),  i = 1, nroots )

  STOP

   A( 0 ) = 3.0_wp / 16.0_wp
   A( 1 ) = - 1.0_wp
   A( 2 ) = 19.0_wp / 16.0_wp
   A( 3 ) = - 1.0_wp
   A( 4 ) = 1.0_wp
!   root = ROOTS_smallest_root_in_interval( A( : 4 ), 0.0_wp, 99.0_wp,         &
!                                           data, control, inform )
!   root = ROOTS_smallest_root_in_interval( A( : 4 ), 0.0_wp, 1.0_wp,          &
!                                           data, control, inform )
!   root = ROOTS_smallest_root_in_interval( A( : 4 ), 0.0_wp, 0.5_wp,          &
!                                           data, control, inform )
!   root = ROOTS_smallest_root_in_interval( A( : 4 ), 0.0_wp, 0.25_wp,         &
!                                           data, control, inform )
!   root = ROOTS_smallest_root_in_interval( A( : 4 ), 0.0_wp, 0.24_wp,         &
!                                           data, control, inform )

   write(6,*) ' new polynomial'
   A( 0 ) = 12.0_wp
   A( 1 ) = - 19.0_wp
   A( 2 ) = 152.0_wp
   A( 3 ) = - 85.0_wp
   A( 4 ) = 12.0_wp

   A( 0 ) = 13.9998500000000003    
   A( 1 ) = -27.7425571428571445   
   A( 2 ) = 13.8712785714285669    
   A( 3 ) = 98.1705519120259282    
   A( 4 ) = -3.2306789871832129E+03

!  root = ROOTS_smallest_root_in_interval( A( : 4 ), 0.0_wp, 1.0_wp,           &
!                                          data, control, inform )

   C( 0 ) =      1.399985000000000d+01
   C( 1 ) =      -2.774255714285714d+0
   C( 2 ) =      1.387127857142857d+01
   C( 3 ) =      5.684341886080801d-14
   C( 4 ) =                     0.0d+0
   C( 5 ) =     -3.637978807091713d-12
   C( 6 ) =     -1.002948227551986d+05
   C( 7 ) =      1.941988380079041d+06
   C( 8 ) =      8.622159565741234d+06
   C( 9 ) =     -1.118517097964874d+07
   C( 10) =     -2.545697005435851d+08
    
!  root = ROOTS_smallest_root_in_interval( C( : 10 ), 0.0_wp, 1.0_wp,          &
!                                          data, control, inform )




   C( 0 ) = 0.9999900000000000     
   C( 1 ) = -1.9799800000000001    
   C( 2 ) = 0.9899900000000003     
   C( 3 ) = -3.5527136788005009D-15
   C( 4 ) = -82.8199999999999932   
   C( 5 ) = 48.4799999999999969    
   C( 6 ) = -0.0000000000000000    

   control%print_level = 1
!  root = ROOTS_smallest_root_in_interval( C( : 6 ), 0.0_wp, 1.0_wp,           &
!                                           data, control, inform )

   control%print_level = 1

   C( 0 ) =  4.6001007573973223D-05
   C( 1 ) = -9.1378939414425107D-05
   C( 2 ) = 4.5689469707212553D-05 
   C( 3 ) = -1.2054307949484545D-26
   C( 4 ) = 1.5879877946159411D-26 
   C( 5 ) = -7.6565262233480169D-15
   C( 6 ) = 1.2753251957596202D-15 
   C( 7 ) = 3.8157744199094053D-20 
   C( 8 ) = -4.7685466292271478D-21
  
   
!  root = ROOTS_smallest_root_in_interval( C( : 8 ), 0.0_wp, 1.0_wp,           &
!                                          data, control, inform )
 
   C( 0 )  =  4.6001007573973223D-05
   C( 1 )  = -9.1378939414425107D-05
   C( 2 )  =  4.5689469707212553D-05
   C( 3 )  = -1.2054307949484545D-26
   C( 4 )  =  1.5879877946159411D-26
   C( 5 )  =  4.6315356177588132D-31
   C( 6 )  =  1.2741129801930031D-15
   C( 7 )  =  6.4425990602716826D-19
   C( 8 )  = -4.7629248876470493D-21
   C( 9 )  = -1.4057616508076847D-24
   C( 10 ) =  9.7861619737717144D-29
  
   C = C * 100000.0_wp

!  root = ROOTS_smallest_root_in_interval( C( : 10 ), 0.0_wp, 1.0_wp,          &
!                                          data, control, inform )

   C( 0 )  =  1.4031350860648831D-14 
   C( 1 )  = -2.8062699917533805D-14 
   C( 2 )  =  1.4031349958766903D-14 
   C( 3 )  =  7.3296544653364487D-29 
   C( 4 )  = -1.8324135423783952D-29

!   C = C * 1.0D+14

!   root = ROOTS_smallest_root_in_interval( C( : 4 ),                          &
!                                           10000.0_wp * EPSILON( one ),       &
!                                           1.0_wp, data, control, inform )
   
   C( 0 )  =  0.0000000000000000    
   C( 1 )  =  2.5664156402392800D-40
   C( 2 )  = -1.3573378690993283D-40
   C( 3 )  = -4.4771807019943293D-35
   C( 4 )  =  0.0000000000000000    
   C( 5 )  =  9.8454244829941588D-23
   C( 6 )  = -1.0000000000000000    
   C( 7 )  =  0.5000001550911866    
   C( 8 )  =  3.2339636665708300D-04
 
!  root = ROOTS_smallest_root_in_interval( C( : 8 ), 0.0_wp, 1.0_wp,           &
!                                          data, control, inform )


   C( 0 )  =  1.0000000000000000_wp 
   C( 1 )  = -1.8099983249837461_wp    
   C( 2 )  =  0.9049991624918756_wp    
   C( 3 )  = -1.6061007101261774D+02
   C( 4 )  = -10.2976422347191487_wp 


!  root = ROOTS_smallest_root_in_interval( C( : 4 ), 0.0_wp,                   &
!                                          1.6478886266254646D-01,             &
!                                          data, control, inform )

  
   C( 0 )  =  1.000000000000000    
   C( 1 )  = -1.995527861434602    
   C( 2 )  =  0.9976316180389976   
   C( 3 )  =  0.000000000000000D+00
   C( 4 )  =  0.000000000000000D+00
   C( 5 )  =  6871939931966449.0D+0
   C( 6 )  = -3.130061193915647D+37
   C( 7 )  =  1.565031036695944D+37

!  root = ROOTS_smallest_root_in_interval( C( : 7 ), 0.0_wp, 1.0_wp,           &
!                                          data, control, inform )


   C( 0 )  =  1.000000000000000      
   C( 1 )  = -1.987483478083446      
   C( 2 )  =  0.9937417390417229     
   C( 3 )  = -0.1252687606231657     
   C( 4 )  =  1.0394758439473678D-009
  
!  root = ROOTS_smallest_root_in_interval( C( : 4 ), 0.0_wp,                   &
!                                          0.5901536777611864_wp,              &
!                                          data, control, inform )

   C( 0 )  =  1.1701792629396736D-07
   C( 1 )  =  4.4546314140434852D+06
   C( 2 )  = -2.2273157070217421D+06
   C( 3 )  = -1.0041943416639333D+09
   C( 4 )  =  4.8511125659889323D+08
  
!control%tol= 0.01_wp * EPSILON( 1.0_wp )
!  root = ROOTS_smallest_root_in_interval( C( : 4 ), 0.0_wp,                   &
!                                          6.9755674488495456D-02,             &
!                                          data, control, inform )
  
 
!6.6564739621803343E-02
!6.6564739620708482E-02
!6.6564739620708607E-02
!6.6564739620708469E-02

!  0.20363632199488485

   C( 0 )  =  5.48616892314348075D-009 
   C( 1 )  =  227695.04922747405       
   C( 2 )  = -113847.52461373706       
   C( 3 )  = -5929792.0411772206       
   C( 4 )  =  4900748.1323242066       
  
   control%tol= EPSILON( 1.0_wp )
!  root = ROOTS_smallest_root_in_interval( C( : 4 ), 0.0_wp,                   &
!                                          0.38439175082608096_wp,             &
!                                          data, control, inform )
    
   C( 0 )  =  3.63319264314644384D-006
   C( 1 )  = -7.24275118845932214D-006
   C( 2 )  =  3.62137559422966065D-006
   C( 3 )  = -8.53334468831281683D-040
   C( 4 )  =  3.17637355220362627D-022
   C( 5 )  = -1.54304412704631573D-006
   C( 6 )  = -5.10812503704212690D-007
   C( 7 )  =  1.10124111934284412D-006
   C( 8 )  = -1.89522385486533516D-007

   root = ROOTS_smallest_root_in_interval( C( : 8 ), 0.0_wp,                   &
                                           0.77445934801417005_wp,             &
                                           data, control, inform )
  
  
   C( 0 )  =  3.07289993024702536D-002
   C( 1 )  = -6.13001374445687058D-002
   C( 2 )  =  3.06500687222843494D-002
   C( 3 )  = -2.93813635446576460D-002
   C( 4 )  = -6.98085144990713297D-003

   root = ROOTS_smallest_root_in_interval( C( : 4 ), 0.0_wp,                   &
                                           0.56279186164331585_wp,             &
                                           data, control, inform )
!  0.56545309528207488     

   write(6, "( ' root = ', ES24.16 )" ) root
   STOP

   WRITE( 6, "( /, ' General tests ' )" )

10 CONTINUE

   DO order = 2, 4
     DO type = 1, 2
       IF ( order == 2 ) THEN
         A( 0 ) = 2.01_wp
         A( 1 ) = - 3.01_wp
         A( 2 ) = 1.01_wp
         IF ( type == 2 ) A( 2 ) = 0.0_wp
       ELSE IF ( order == 3 ) THEN
         A( 0 ) = - 6.01_wp
         A( 1 ) = 11.01_wp
         A( 2 ) = -6.01_wp
         A( 3 ) = 1.01_wp
         IF ( type == 2 ) A( 3 ) = 0.0_wp
       ELSE
         IF ( type == 1 ) THEN
           A( 0 ) = 24.001_wp
           A( 1 ) = -50.001_wp
           A( 2 ) = 35.001_wp
           A( 3 ) = -10.001_wp
           A( 4 ) = 1.001_wp
         ELSE
           A( 0 ) = 1.00_wp
           A( 1 ) = -4.00_wp
           A( 2 ) = 6.00_wp
           A( 3 ) = -4.00_wp
           A( 4 ) = 1.00_wp
         END IF
       END IF

       IF ( type == 1 ) THEN
         IF ( order == 2 ) THEN
           WRITE( 6, "( /, ' Quadratic ' )" )
           CALL ROOTS_quadratic( A( 0 ), A( 1 ), A( 2 ), control%tol,          &
             nroots, ROOTS( 1 ), ROOTS( 2 ), debug )
         ELSE IF ( order == 3 ) THEN
           WRITE( 6, "( /, ' Cubic ' )" )
           CALL ROOTS_cubic( A( 0 ), A( 1 ), A( 2 ), A( 3 ), control%tol,      &
             nroots, ROOTS( 1 ), ROOTS( 2 ), ROOTS( 3 ), debug )
         ELSE
           WRITE( 6, "( /, ' Quartic ' )" )
           CALL ROOTS_quartic( A( 0 ), A( 1 ), A( 2 ), A( 3 ), A( 4 ),         &
             control%tol, nroots, ROOTS( 1 ), ROOTS( 2 ), ROOTS( 3 ),          &
             ROOTS( 4 ), debug )
         END IF
       ELSE
         IF ( order == 2 ) THEN
           WRITE( 6, "( /, ' Quadratic ' )" )
           CALL ROOTS_solve( A( 0 : order ), nroots, ROOTS( : order ),         &
                             control, inform, data )
         ELSE IF ( order == 3 ) THEN
           WRITE( 6, "( /, ' Cubic ' )" )
           CALL ROOTS_solve( A( 0 : order ), nroots, ROOTS( : order ),         &
                             control, inform, data )
         ELSE
           WRITE( 6, "( /, ' Quartic ' )" )
           CALL ROOTS_solve( A( 0 : order ), nroots, ROOTS( : order ),         &
                             control, inform, data )
         END IF
       END IF
       IF ( nroots == 0 ) THEN
         WRITE( 6, "( ' no real roots ' )" )
       ELSE IF ( nroots == 1 ) THEN
         WRITE( 6, "( ' 1 real root ' )" )
       ELSE IF ( nroots == 2 ) THEN
         WRITE( 6, "( ' 2 real roots ' )" )
       ELSE IF ( nroots == 3 ) THEN
         WRITE( 6, "( ' 3 real roots ' )" )
       ELSE IF ( nroots == 4 ) THEN
         WRITE( 6, "( ' 4 real roots ' )" )
       END IF
       IF ( nroots /= 0 ) WRITE( 6, "( ' roots: ', 4ES12.4 )") ROOTS( : nroots )
       IF ( nroots /= 0 ) WRITE( 6, "( ' value: ', 4ES12.4 )")                 &
        ( ROOTS_polynomial_value( ROOTS( i ), A( 0 : order ) ),  i = 1, nroots )
       
     END DO
   END DO

!  Test for error exits

   WRITE(6,"( /, ' Tests for error exits ' )" )
   A1( 0 : 4 ) = A ; A1( 5 ) = 1.0_wp
   CALL ROOTS_solve( A1, nroots, ROOTS, control, inform, data )
   WRITE(6,"( ' Test 3: exit status ', I0 )" ) inform%status
   A2 = A(  0 : 2 )
   CALL ROOTS_solve( A2, nroots, ROOTS2, control, inform, data )
   WRITE(6,"( ' Test 4: exit status ', I0 )" ) inform%status

   STOP

   END PROGRAM GALAHAD_ROOTS_test_deck

