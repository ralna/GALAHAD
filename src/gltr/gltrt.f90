! THIS VERSION: GALAHAD 2.1 - 22/03/2007 AT 09:00 GMT.
   PROGRAM GALAHAD_GLTR_test_deck
   USE GALAHAD_GLTR_DOUBLE                            ! double precision version
   IMPLICIT NONE
   INTEGER, PARAMETER :: working = KIND( 1.0D+0 ) ! set precision
   REAL ( KIND = working ), PARAMETER :: one = 1.0_working, two = 2.0_working
   INTEGER, PARAMETER :: n = 100                  ! problem dimension
   INTEGER :: i, nn, pass
   REAL ( KIND = working ) :: f, radius
   REAL ( KIND = working ), DIMENSION( n ) :: X, R, VECTOR, H_vector
   REAL ( KIND = working ), DIMENSION( 0 ) :: X0, R0, VECTOR0
   TYPE ( GLTR_data_type ) :: data
   TYPE ( GLTR_control_type ) :: control        
   TYPE ( GLTR_info_type ) :: info

!  ==============
!  Normal entries
!  ==============

   WRITE( 6, "( /, ' ==== normal exits ====== ', / )" )

! Initialize control parameters

   OPEN( UNIT = 23, STATUS = 'SCRATCH' )
!  OPEN( UNIT = 23 )
   DO pass = 1, 11
      IF ( pass /= 4 .AND. pass /= 7 .AND. pass /= 8 )                         &
           CALL GLTR_initialize( data, control, info )
      control%error = 23 ; control%out = 23 ; control%print_level = 10
      info%status = 1
      radius = one
      IF ( pass == 2 ) control%unitm = .FALSE. ; radius = 1000.0_working
      IF ( pass == 4 ) THEN
           radius = radius / two ; info%status = 4
      END IF           
      IF ( pass == 5 ) radius = 0.0001_working
      IF ( pass == 7 ) THEN
         radius = 0.1_working ; control%boundary = .TRUE. ; info%status = 4
      END IF
      IF ( pass == 8 ) THEN
         radius = 100.0_working ; info%status = 4
      END IF
      IF ( pass == 9 ) radius = 10.0_working
      IF ( pass == 10 ) radius = 10.0_working
      IF ( pass == 11 ) radius = 10000.0_working

      IF ( pass == 10 .OR. pass == 11 ) THEN
         R( : n - 1 ) = 0.000000000001_working ; R( n ) = one
      ELSE
         R = one
      END IF

!  Iteration to find the minimizer

      DO                                     
         CALL GLTR_solve( n, radius, f, X, R, VECTOR, data, control, info )

! Branch as a result of info%status

         SELECT CASE( info%status )

!  Form the preconditioned gradient

         CASE( 2, 6 )                  
            VECTOR = VECTOR / two

!  Form the matrix-vector product

         CASE ( 3, 7 )                 
            IF ( pass == 2 .OR. pass == 6 .OR. pass == 7 .OR. pass == 8 ) THEN
               H_vector( 1 ) =  two * VECTOR( 1 ) + VECTOR( 2 )
               DO i = 2, n - 1
                 H_vector( i ) = VECTOR( i - 1 ) + two * VECTOR( i ) +         &
                                 VECTOR( i + 1 )
               END DO
               H_vector( n ) = VECTOR( n - 1 ) + two * VECTOR( n )
            ELSE IF ( pass == 9 ) THEN
               H_vector( 1 ) = VECTOR( 1 ) + VECTOR( 2 )
               DO i = 2, n - 1
                 H_vector( i ) = VECTOR( i - 1 ) - two * VECTOR( i ) +         &
                                 VECTOR( i + 1 )
               END DO
               H_vector( n ) = VECTOR( n - 1 ) + VECTOR( n )
            ELSE IF ( pass == 10 .OR. pass == 11 ) THEN
              H_vector( 1 ) = - two * VECTOR( 1 )
              H_vector( 2 : n - 1 ) = 0.0001_working * VECTOR( 2 : n - 1 )
              H_vector( n ) = - VECTOR( n )
            ELSE
               H_vector( 1 ) = - two * VECTOR( 1 ) + VECTOR( 2 )
               DO i = 2, n - 1
                 H_vector( i ) = VECTOR( i - 1 ) - two * VECTOR( i ) +         &
                                 VECTOR( i + 1 )
               END DO
               H_vector( n ) = VECTOR( n - 1 ) - two * VECTOR( n )
            END IF
            VECTOR = H_vector 

!  Restart

         CASE ( 5 )       
            IF ( pass == 10 .OR. pass == 11 ) THEN
               R( : n - 1 ) = 0.000000000001_working
               R( n ) = one
            ELSE
               R = one
            END IF

!  Successful return

         CASE ( - 2 : 0 )  
            EXIT

!  Error returns

         CASE DEFAULT      
            EXIT
         END SELECT
      END DO

      WRITE( 6, "( ' pass ', I3, ' GLTR_solve exit status = ', I6 )" )         &
             pass, info%status
!     WRITE( 6, "( ' its, solution and Lagrange multiplier = ', I6, 2ES12.4 )")&
!               info%iter + info%iter_pass2, f, info%multiplier
      IF ( pass /= 3 .AND. pass /= 6 .AND. pass /= 7 )                         &
        CALL GLTR_terminate( data, control, info ) !  delete internal workspace
   END DO

!  =============
!  Error entries
!  =============

   WRITE( 6, "( /, ' ==== error exits ====== ', / )" )

! Initialize control parameters

   DO pass = 1, 5
      radius = one
      CALL GLTR_initialize( data, control, info )
      control%error = 23 ; control%out = 23 ; control%print_level = 10
      info%status = 1
      nn = n
      IF ( pass == 1 ) control%steihaug_toint = .TRUE.
      IF ( pass == 2 ) control%itmax = 0
      IF ( pass == 3 ) control%unitm = .FALSE.
      IF ( pass == 4 ) nn = 0
      IF ( pass == 5 ) radius = - one

      R = one

!  Iteration to find the minimizer

      DO                                     
         IF ( pass /= 4 ) THEN
           CALL GLTR_solve( nn, radius, f, X( : nn ), R( : nn ),                &
                            VECTOR( : nn ), data, control, info )
         ELSE
           CALL GLTR_solve( nn, radius, f, X0, R0,                              &
                            VECTOR0, data, control, info )
         END IF

! Branch as a result of info%status

         SELECT CASE( info%status )

!  Form the preconditioned gradient

         CASE( 2, 6 )                  
            IF ( pass /= 3 ) THEN
               VECTOR = VECTOR / two
            ELSE
               VECTOR = - VECTOR / two
            END IF

!  Form the matrix-vector product

         CASE ( 3, 7 )                 
            H_vector( 1 ) = - two * VECTOR( 1 ) + VECTOR( 2 )
            DO i = 2, n - 1
              H_vector( i ) = VECTOR( i - 1 ) - two * VECTOR( i ) +         &
                              VECTOR( i + 1 )
            END DO
            H_vector( n ) = VECTOR( n - 1 ) - two * VECTOR( n )
            VECTOR = H_vector 

!  Restart

         CASE ( 5 )       
            R = one

!  Successful return

         CASE ( - 2 : 0 )  
            EXIT

!  Error returns

         CASE DEFAULT      
            EXIT
         END SELECT
      END DO

      WRITE( 6, "( ' pass ', I3, ' GLTR_solve exit status = ', I6 )" )         &
             pass, info%status
      CALL GLTR_terminate( data, control, info ) !  delete internal workspace
   END DO
   CLOSE( unit = 23 )

!  STOP
   END PROGRAM GALAHAD_GLTR_test_deck
