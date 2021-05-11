! THIS VERSION: GALAHAD 2.6 - 24/02/2014 AT 10:10 GMT.
   PROGRAM GALAHAD_LLST_test_deck
   USE GALAHAD_LLST_DOUBLE                            ! double precision version
   IMPLICIT NONE
   INTEGER, PARAMETER :: working = KIND( 1.0D+0 ) ! set precision
   REAL ( KIND = working ), PARAMETER :: one = 1.0_working, zero = 0.0_working
   INTEGER, PARAMETER :: m = 5000, n = 2 * m + 1   ! problem dimensions
   INTEGER :: i, pass, problem, nn
   REAL ( KIND = working ), DIMENSION( n ) :: X
   REAL ( KIND = working ), DIMENSION( m ) :: B
   REAL ( KIND = working ) :: radius

   TYPE ( LLST_data_type ) :: data
   TYPE ( LLST_control_type ) :: control
   TYPE ( LLST_inform_type ) :: inform
   TYPE ( SMT_type ) :: A, S

   B = one                               ! The term b is a vector of ones
   A%m = m ; A%n = n ; A%ne = m          ! A^T = ( I : Diag(1:n) )
   CALL SMT_put( A%type, 'COORDINATE', i )
   ALLOCATE( A%row( 3 * m ), A%col( 3 * m ), A%val( 3 * m ) )
   DO i = 1, m
     A%row( i ) = i ; A%col( i ) = i ; A%val( i ) = one
     A%row( m + i ) = i ; A%col( m + i ) = m + i
     A%val( m + i ) = REAL( i, working )
     A%row( 2 * m + i ) = i ; A%col( 2 * m + i ) = n
     A%val( 2 * m + i ) = one
   END DO
   S%m = n ; S%n = n ; S%ne = n    ! S = diag(1:n)**2
   CALL SMT_put( S%type, 'DIAGONAL', i )
   ALLOCATE( S%val( n ) )
   DO i = 1, n
     S%val( i ) = REAL( i * i, working )
   END DO

!  ==============
!  Normal entries
!  ==============

   WRITE( 6, "( /, ' ==== normal exits ====== ', / )" )

! Initialize control parameters

   OPEN( UNIT = 23, STATUS = 'SCRATCH' )
   DO problem = 1, 2
     DO pass = 1, 5
       CALL LLST_initialize( data, control, inform )
!      control%print_level = 1
!      control%itmax = 50
!      control%extra_vectors = 100
       control%error = 23 ; control%out = 23 ; control%print_level = 10
       radius = one
       IF ( pass == 2 ) radius = 10.0_working
       IF ( pass == 3 ) radius = 0.0001_working
       IF ( pass == 4 ) THEN
         inform%status = 5
         radius = 0.0001_working
         control%prefix = '"LLST: "     '
!        control%error = 6 ; control%out = 6 ; control%print_level = 1
!        radius = 10.0_working
       END IF
       IF ( pass == 5 ) THEN
!        if(problem==2)stop
         control%prefix = '"LLST: "     '
!        control%error = 6 ; control%out = 6 ; control%print_level = 1
!        radius = 10.0_working
       END IF

       IF ( problem == 1 ) THEN
         CALL LLST_solve( m, n, radius, A, B, X, data, control, inform )
       ELSE
         CALL LLST_solve( m, n, radius, A, B, X, data, control, inform, S = S )
       END IF
       WRITE( 6, "( ' problem ', I1, ' pass = ', I1,                           &
      &  ' LLST_solve exit status = ', I6 )" ) problem, pass, inform%status
       CALL LLST_terminate( data, control, inform ) ! delete workspace
     END DO
   END DO

!  =============
!  Error entries
!  =============

   WRITE( 6, "( /, ' ==== error exits ====== ', / )" )

! Initialize control parameters

   DO pass = 1, 6
      radius = one
      CALL LLST_initialize( data, control, inform )
      control%error = 23 ; control%out = 23 ; control%print_level = 10
      IF ( pass == 1 ) nn = 0
      IF ( pass == 2 ) radius = - one
      IF ( pass == 3 ) CALL SMT_put( A%type, 'UNCOORDINATE', i )
      IF ( pass == 4 ) CALL SMT_put( S%type, 'UNDIAGONAL', i )
!      IF ( pass == 1 ) control%equality_problem = .TRUE.
      IF ( pass == 5 ) THEN
        DO i = 1, n
          S%val( i ) = - REAL( i * i, working )
        END DO
      END IF
      IF ( pass == 6 ) THEN
        control%max_factorizations = 1
        radius = 100.0_working
      END IF

!  Iteration to find the minimizer

      IF ( pass /= 1 ) THEN
        CALL LLST_solve( m, n, radius, A, B, X, data, control, inform, S = S )
      ELSE
        CALL LLST_solve( 0, nn, radius, A, B, X, data, control, inform )
      END IF
      IF ( pass == 3 ) CALL SMT_put( A%type, 'COORDINATE', i )
      IF ( pass == 4 ) CALL SMT_put( S%type, 'DIAGONAL', i )
      IF ( pass == 5 ) THEN
        DO i = 1, n
          S%val( i ) = REAL( i * i, working )
        END DO
      END IF
      WRITE( 6, "( ' pass ', I3, ' LLST_solve exit status = ', I6 )" )         &
             pass, inform%status
      CALL LLST_terminate( data, control, inform ) !  delete internal workspace
   END DO
   CLOSE( unit = 23 )

   END PROGRAM GALAHAD_LLST_test_deck
