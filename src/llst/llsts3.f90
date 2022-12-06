   PROGRAM GALAHAD_LLST_EXAMPLE  !  GALAHAD 2.6 - 24/02/2014 AT 09:50 GMT
   USE GALAHAD_LLST_DOUBLE                         ! double precision version
   IMPLICIT NONE
   INTEGER, PARAMETER :: working = KIND( 1.0D+0 )  ! set precision
   REAL ( KIND = working ), PARAMETER :: one = 1.0_working, zero = 0.0_working
   INTEGER, PARAMETER :: n = 50, m = 2 * n         ! problem dimensions
   INTEGER :: i, l
   REAL ( KIND = working ) :: radius = 10.0_working ! radius of one
   REAL ( KIND = working ), DIMENSION( n ) :: X
   REAL ( KIND = working ), DIMENSION( m ) :: B
   TYPE ( SMT_type ) :: A, S
   TYPE ( LLST_data_type ) :: data
   TYPE ( LLST_control_type ) :: control
   TYPE ( LLST_inform_type ) :: inform
   CALL LLST_initialize( data, control, inform ) ! Initialize control parameters
   control%print_level = 1
   control%max_factorizations = 100
!  control%sbls_control%print_level = 1
   control%equality_problem = .TRUE.
   B = one                               ! The term b is a vector of ones
   A%m = m ; A%n = n ; A%ne = m          ! A^T = ( I : Diag(1:n) )
   CALL SMT_put( A%type, 'COORDINATE', i )
   ALLOCATE( A%row( m ), A%col( m ), A%val( m ) )
   DO i = 1, n
     A%row( i ) = i ; A%col( i ) = i ; A%val( i ) = one
     A%row( n + i ) = n + i ; A%col( n + i ) = i
     A%val( n + i ) = REAL( i, working )
   END DO
   S%m = n ; S%n = n ; S%ne = n    ! S = diag(1:n)**2
   CALL SMT_put( S%type, 'DIAGONAL', i )
   ALLOCATE( S%val( n ) )
   DO i = 1, n
     S%val( i ) = REAL( i * i, working )
   END DO
   CALL LLST_solve( m, n, radius, A, B, X, data, control, inform, S = S )
   IF ( inform%status == 0 ) THEN  !  Successful return
     DO l = 1, A%ne
       i =  A%row( l )
       B( i ) = B( i ) - A%val( l ) * X( A%col( l ) )
     END DO
     WRITE( 6, "( '   ||x||_2  recurred and calculated = ', 2ES16.8 )" )       &
       inform%x_norm, SQRT( DOT_PRODUCT( X, S%val * X ) )
     WRITE( 6, "( ' ||Ax-b||_S recurred and calculated = ', 2ES16.8 )" )       &
       inform%r_norm, SQRT( DOT_PRODUCT( B, B ) )
   ELSE                            !  Error returns
     WRITE( 6, "( ' LLST_solve exit status = ', I0 ) " ) inform%status
   END IF
   CALL LLST_terminate( data, control, inform ) ! delete workspace
   DEALLOCATE( A%row, A%col, A%val, S%val, A%type, S%type )
   END PROGRAM GALAHAD_LLST_EXAMPLE
