   PROGRAM GALAHAD_LLSR_EXAMPLE  !  GALAHAD 4.1 - 2023-06-05 AT 13:15 GMT
   USE GALAHAD_LLSR_DOUBLE                         ! double precision version
   IMPLICIT NONE
   INTEGER, PARAMETER :: working = KIND( 1.0D+0 )  ! set precision
   REAL ( KIND = working ), PARAMETER :: one = 1.0_working, zero = 0.0_working
   INTEGER, PARAMETER :: m = 1000, n = 2 * m + 1   ! problem dimensions
   INTEGER :: i, l
   REAL ( KIND = working ) :: power = 3.0_working ! cubic regularization
   REAL ( KIND = working ) :: weight = 0.1_working ! weight of 1/10th
   REAL ( KIND = working ), DIMENSION( n ) :: X
   REAL ( KIND = working ), DIMENSION( m ) :: B
   TYPE ( SMT_type ) :: A, S
   TYPE ( LLSR_data_type ) :: data
   TYPE ( LLSR_control_type ) :: control
   TYPE ( LLSR_inform_type ) :: inform
   CALL LLSR_initialize( data, control, inform ) ! Initialize control parameters
   control%sbls_control%symmetric_linear_solver = "sytr  "
   control%sbls_control%definite_linear_solver = "sytr  "
   B = one                               ! The term b is a vector e of ones
   A%m = m ; A%n = n ; A%ne = 3 * m      ! A = ( I : Diag(1:n) : e)
   CALL SMT_put( A%type, 'COORDINATE', i )
   ALLOCATE( A%row( A%ne ), A%col( A%ne ), A%val( A%ne ) )
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
   CALL LLSR_solve( m, n, power, weight, A, B, X, data, control, inform, S = S )
   IF ( inform%status == 0 ) THEN  !  Successful return
     DO l = 1, A%ne
       i =  A%row( l )
       B( i ) = B( i ) - A%val( l ) * X( A%col( l ) )
     END DO
     WRITE( 6, "( '   ||x||_S  recurred and calculated = ', 2ES16.8 )" )       &
       inform%x_norm, SQRT( DOT_PRODUCT( X, S%val * X ) )
     WRITE( 6, "( ' ||Ax-b||_2 recurred and calculated = ', 2ES16.8 )" )       &
       inform%r_norm, SQRT( DOT_PRODUCT( B, B ) )
   ELSE                            !  Error returns
     WRITE( 6, "( ' LLSR_solve exit status = ', I0 ) " ) inform%status
   END IF
   CALL LLSR_terminate( data, control, inform ) ! delete workspace
   DEALLOCATE( A%row, A%col, A%val, S%val, A%type, S%type )
   END PROGRAM GALAHAD_LLSR_EXAMPLE
