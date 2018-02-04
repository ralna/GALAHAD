! THIS VERSION: GALAHAD 2.1 - 22/03/2007 AT 09:00 GMT.
   PROGRAM GALAHAD_LSRT_test_deck
   USE GALAHAD_LSRT_DOUBLE                            ! double precision version
   IMPLICIT NONE
   INTEGER, PARAMETER :: working = KIND( 1.0D+0 )     ! set precision
   REAL ( KIND = working ), PARAMETER :: one = 1.0_working, zero = 0.0_working
   INTEGER, PARAMETER :: n = 50, m = 2 * n            ! problem dimensions
   INTEGER, PARAMETER :: m2 = 50, n2 = 2 * m2         ! 2nd problem dimensions
   INTEGER :: i, pass, problem, nn
   REAL ( KIND = working ), DIMENSION( n ) :: X, V
   REAL ( KIND = working ), DIMENSION( m ) :: U
   REAL ( KIND = working ), DIMENSION( n2 ) :: X2, V2
   REAL ( KIND = working ), DIMENSION( m2 ) :: U2
   REAL ( KIND = working ), DIMENSION( 0 ) :: X0, V0, U0
   REAL ( KIND = working ) :: p, sigma

   TYPE ( LSRT_data_type ) :: data
   TYPE ( LSRT_control_type ) :: control
   TYPE ( LSRT_inform_type ) :: inform

!  ==============
!  Normal entries
!  ==============

   WRITE( 6, "( /, ' ==== normal exits ====== ', / )" )

! Initialize control parameters

   OPEN( UNIT = 23, STATUS = 'SCRATCH' )
   DO problem = 1, 2
     DO pass = 1, 9
       IF ( pass /= 4 .AND. pass /= 7 .AND. pass /= 8 )                         &
         CALL LSRT_initialize( data, control, inform )
!      control%print_level = 1
!      control%itmax = 50
!      control%extra_vectors = 100
       control%error = 23 ; control%out = 23 ; control%print_level = 10
       inform%status = 1
       sigma = one
       p = 3.0_working
       IF ( pass == 2 ) sigma = 10.0_working
       IF ( pass == 3 ) sigma = 0.0001_working
       IF ( pass == 4 ) THEN
         control%fraction_opt = 0.99_working
       END IF      
       IF ( pass == 5 ) THEN
         control%fraction_opt = 0.99_working
         control%extra_vectors = 1
       END IF      
       IF ( pass == 6 ) THEN
         control%fraction_opt = 0.99_working
         control%extra_vectors = 100
       END IF      
       IF ( pass == 7 ) THEN
         control%fraction_opt = 0.99_working
         control%extra_vectors = 100
       END IF
       IF ( pass == 8 ) THEN
         control%fraction_opt = 0.99_working
         p = 2.0_working
       END IF
       IF ( pass == 9 ) THEN
         control%prefix = '"LSRT: "     '
!        control%error = 6 ; control%out = 6 ; control%print_level = 1
!        sigma = 10.0_working
       END IF

       IF ( problem == 1 ) THEN
         U = one
         DO
           CALL LSRT_solve( m, n, p, sigma, X, U, V, data, control, inform )

           SELECT CASE( inform%status )  ! Branch as a result of inform%status
           CASE( 2 )                     !  Form u <- u + A * v
             U( : n ) = U( : n ) + V
             DO i = 1, n
               U( n + i ) = U( n + i ) + i * V( i )
             END DO
           CASE( 3 )                     !  Form v <- v + A^T * u
             V = V + U( : n )            !  A^T = ( I : diag(1:n) )
             DO i = 1, n
               V( i ) = V( i ) + i * U( n + i )
             END DO
           CASE ( 4 )                    ! Restart
              U = one
           CASE DEFAULT      
              EXIT
           END SELECT
         END DO
       ELSE
         U2 = one
         DO
           CALL LSRT_solve( m2, n2, p, sigma, X2, U2, V2, data, control, inform )

           SELECT CASE( inform%status )  ! Branch as a result of inform%status
           CASE( 2 )                     !  Form u <- u + A * v
             U2 = U2 + V2( : n )         !  A = ( I : diag(1:n) )
             DO i = 1, n
               U2( i ) = U2( i ) + i * V2( n + i )
             END DO
           CASE( 3 )                     !  Form v <- v + A^T * u
             V2( : n ) = V2( : n ) + U2
             DO i = 1, n
               V2( n + i ) = V2( n + i ) + i * U2( i )
             END DO
           CASE ( 4 )                    ! Restart
              U2 = one
           CASE DEFAULT      
              EXIT
           END SELECT
         END DO
       END IF
       WRITE( 6, "( ' problem ', I1, ' pass ', I3,                              &
      &     ' LSRT_solve exit status = ', I6 )" ) problem, pass, inform%status
!      WRITE( 6, "( ' its, solution and Lagrange multiplier = ', I6, 2ES12.4 )")&
!                inform%iter + inform%iter_pass2, f, inform%multiplier
       CALL LSRT_terminate( data, control, inform ) ! delete internal workspace
     END DO
   END DO

!  =============
!  Error entries
!  =============

   WRITE( 6, "( /, ' ==== error exits ====== ', / )" )

! Initialize control parameters

   DO pass = 2, 8
      sigma = one
      p = 3.0_working
      CALL LSRT_initialize( data, control, inform )
      control%error = 23 ; control%out = 23 ; control%print_level = 10
      inform%status = 1
      U = one
      IF ( pass == 2 ) control%itmax = 0
      IF ( pass == 3 ) inform%status = 0
      IF ( pass == 4 ) nn = 0
      IF ( pass == 5 ) sigma = - one
      IF ( pass == 6 .OR. pass == 7 ) CYCLE
      IF ( pass == 8 ) p = one

!  Iteration to find the minimizer

      DO                                     
        IF ( pass /= 4 ) THEN
          CALL LSRT_solve( m, n, p, sigma, X, U, V, data, control, inform )
        ELSE
          CALL LSRT_solve( 0, nn, p, sigma, X0, U0, V0, data, control, inform )
        END IF

        SELECT CASE( inform%status )  ! Branch as a result of inform%status
        CASE( 2 )                     !  Form u <- u + A * v
          U( : n ) = U( : n ) + V
          DO i = 1, n
            U( n + i ) = U( n + i ) + i * V( i )
          END DO
        CASE( 3 )                     !  Form v <- v + A^T * u
          V = V + U( : n )            !  A^T = ( I : diag(1:n) )
          DO i = 1, n
            V( i ) = V( i ) + i * U( n + i )
          END DO
        CASE ( 4 )                    ! Restart
           U = one
        CASE DEFAULT      
           EXIT
        END SELECT
      END DO
      WRITE( 6, "( ' pass ', I3, ' LSRT_solve exit status = ', I6 )" )         &
             pass, inform%status
      CALL LSRT_terminate( data, control, inform ) !  delete internal workspace
   END DO
   CLOSE( unit = 23 )

   END PROGRAM GALAHAD_LSRT_test_deck
