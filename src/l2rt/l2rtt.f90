! THIS VERSION: GALAHAD 2.4 - 08/04/2010 AT 08:00 GMT.
   PROGRAM GALAHAD_L2RT_test_deck
   USE GALAHAD_L2RT_DOUBLE                            ! double precision version
   USE GALAHAD_SYMBOLS
   IMPLICIT NONE
   INTEGER, PARAMETER :: working = KIND( 1.0D+0 )     ! set precision
   REAL ( KIND = working ), PARAMETER :: one = 1.0_working, zero = 0.0_working
   INTEGER, PARAMETER :: n = 50, m = 2 * n            ! problem dimensions
   INTEGER, PARAMETER :: m2 = 50, n2 = 2 * m2         ! 2nd problem dimensions
   INTEGER :: i, pass, status, problem, nn
   REAL ( KIND = working ), DIMENSION( n ) :: X, V
   REAL ( KIND = working ), DIMENSION( m ) :: U
   REAL ( KIND = working ), DIMENSION( n2 ) :: X2, V2
   REAL ( KIND = working ), DIMENSION( m2 ) :: U2
   REAL ( KIND = working ), DIMENSION( 0 ) :: X0, V0, U0
   REAL ( KIND = working ) :: p, sigma, mu

   TYPE ( L2RT_data_type ) :: data
   TYPE ( L2RT_control_type ) :: control
   TYPE ( L2RT_inform_type ) :: inform

!  ==============
!  Normal entries
!  ==============

   WRITE( 6, "( /, ' ==== normal exits ====== ', / )" )

! Initialize control parameters

   OPEN( UNIT = 23, STATUS = 'SCRATCH' )
   DO problem = 1, 2
     DO pass = 1, 10
       IF ( pass /= 4 .AND. pass /= 7 .AND. pass /= 8 )                        &
         CALL L2RT_initialize( data, control, inform )
!      control%print_level = 1
!      control%itmax = 50
!      control%extra_vectors = 100
       control%error = 23 ; control%out = 23 ; control%print_level = 10
       inform%status = 1
       sigma = one
       p = 3.0_working
       mu = 0.0_working
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
         control%prefix = '"L2RT: "     '
!        control%error = 6 ; control%out = 6 ; control%print_level = 1
!        sigma = 10.0_working
       END IF
       IF ( pass == 10 ) THEN
         control%prefix = '"L2RT: "     '
!        control%error = 6 ; control%out = 6 ; control%print_level = 1
         p = 2.0_working
         mu = 1.0_working
       END IF

       IF ( problem == 1 ) THEN
         U = one
!        control%error = 6 ; control%out = 6 ; control%print_level = 1
         DO
           CALL L2RT_solve( m, n, p, sigma, mu, X, U, V, data, control, inform )

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
 !       control%error = 6 ; control%out = 6 ; control%print_level = 1
         DO
           CALL L2RT_solve( m2, n2, p, sigma, mu, X2, U2, V2, data, control,   &
                            inform )

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
      &     ' L2RT_solve exit status = ', I6 )" ) problem, pass, inform%status
!      WRITE( 6, "( ' its, solution and Lagrange multiplier = ', I6, 2ES12.4 )")&
!                inform%iter + inform%iter_pass2, f, inform%multiplier
       CALL L2RT_terminate( data, control, inform ) ! delete internal workspace
     END DO
   END DO

!  =============
!  Error entries
!  =============

   WRITE( 6, "( /, ' ==== error exits ====== ', / )" )

! Initialize control parameters

   status = 3
   DO pass = 1, 4
      sigma = one
      p = 3.0_working
      mu = zero
      CALL L2RT_initialize( data, control, inform )
      control%error = 23 ; control%out = 23 ; control%print_level = 10
      inform%status = 1
      U = one
      IF ( pass == 1 ) nn = 0
      IF ( pass == 2 ) sigma = - one
      IF ( pass == 3 ) p = one
      IF ( pass == 4 ) mu = - one

!  Iteration to find the minimizer

      DO                                     
        IF ( pass == 1 ) THEN
          CALL L2RT_solve( 0, nn, p, sigma, mu, X0, U0, V0, data, control,     &
                           inform )
        ELSE
          CALL L2RT_solve( m, n, p, sigma, mu, X, U, V, data, control, inform )
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
      WRITE( 6, "(  I3, ' L2RT_solve exit status = ', I6 )" )         &
             status, inform%status
      CALL L2RT_terminate( data, control, inform ) !  delete internal workspace
   END DO

   DO status = 1, 25
     IF ( status == - GALAHAD_error_allocate ) CYCLE
     IF ( status == - GALAHAD_error_deallocate ) CYCLE
     IF ( status == - GALAHAD_error_restrictions ) CYCLE
     IF ( status == - GALAHAD_error_bad_bounds ) CYCLE
     IF ( status == - GALAHAD_error_primal_infeasible ) CYCLE
     IF ( status == - GALAHAD_error_dual_infeasible ) CYCLE
     IF ( status == - GALAHAD_error_unbounded ) CYCLE
     IF ( status == - GALAHAD_error_no_center ) CYCLE
     IF ( status == - GALAHAD_error_analysis ) CYCLE
     IF ( status == - GALAHAD_error_factorization ) CYCLE
     IF ( status == - GALAHAD_error_solve ) CYCLE
     IF ( status == - GALAHAD_error_uls_analysis ) CYCLE
     IF ( status == - GALAHAD_error_uls_factorization ) CYCLE
     IF ( status == - GALAHAD_error_uls_solve ) CYCLE
     IF ( status == - GALAHAD_error_preconditioner ) CYCLE
     IF ( status == - GALAHAD_error_ill_conditioned ) CYCLE
     IF ( status == - GALAHAD_error_tiny_step ) CYCLE
!    IF ( status == - GALAHAD_error_max_iterations ) CYCLE
     IF ( status == - GALAHAD_error_cpu_limit ) CYCLE
     IF ( status == - GALAHAD_error_inertia ) CYCLE
     IF ( status == - GALAHAD_error_file ) CYCLE
     IF ( status == - GALAHAD_error_io ) CYCLE
     IF ( status == - GALAHAD_error_upper_entry ) CYCLE
     IF ( status == - GALAHAD_error_sort ) CYCLE
     IF ( status == - GALAHAD_error_sort ) CYCLE
!    IF ( status == - GALAHAD_error_input_status ) CYCLE

      sigma = one
      p = 3.0_working
      mu = zero
      CALL L2RT_initialize( data, control, inform )
      control%error = 23 ; control%out = 23 ; control%print_level = 10
      inform%status = 1
      U = one
      IF ( status == - GALAHAD_error_max_iterations ) THEN
        control%itmax = 0
      ELSE IF ( status == - GALAHAD_error_input_status ) THEN
        inform%status = 0
      ELSE
        EXIT
      END IF

!  Iteration to find the minimizer

      DO                                     
        CALL L2RT_solve( m, n, p, sigma, mu, X, U, V, data, control, inform )

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
      WRITE( 6, "(  I3, ' L2RT_solve exit status = ', I6 )" )         &
             status, inform%status
      CALL L2RT_terminate( data, control, inform ) !  delete internal workspace
   END DO
   CLOSE( unit = 23 )

   END PROGRAM GALAHAD_L2RT_test_deck
