   PROGRAM GALAHAD_LLSR2_EXAMPLE  !  GALAHAD 4.1 - 2023-06-05 AT 13:15 GMT
   USE GALAHAD_LLSR_DOUBLE                         ! double precision version
   IMPLICIT NONE
   INTEGER, PARAMETER :: working = KIND( 1.0D+0 )  ! set precision
   REAL ( KIND = working ), PARAMETER :: one = 1.0_working, zero = 0.0_working
   INTEGER, PARAMETER :: m = 50, n = 2 * m         ! problem dimensions
   INTEGER :: i, l
   REAL ( KIND = working ) :: power = 3.0_working ! cubic regularization
   REAL ( KIND = working ) :: weight = 0.1_working ! weigth of 1/10th
   REAL ( KIND = working ), DIMENSION( n ) :: X
   REAL ( KIND = working ), DIMENSION( m ) :: B
   TYPE ( SMT_type ) :: A, S
   TYPE ( LLSR_data_type ) :: data
   TYPE ( LLSR_control_type ) :: control
   TYPE ( LLSR_inform_type ) :: inform
   CALL LLSR_initialize( data, control, inform ) ! Initialize control parameters
   control%print_level = 1
   B = one                               ! The term b is a vector of ones
   A%m = m ; A%n = n ; A%ne = m          ! A = ( I : Diag(1:m) )
   CALL SMT_put( A%type, 'COORDINATE', i )
   ALLOCATE( A%row( n ), A%col( n ), A%val( n ) )
   DO i = 1, m
     A%row( i ) = i ; A%col( i ) = i ; A%val( i ) = one
     A%row( m + i ) = i ; A%col( m + i ) = m + i
     A%val( m + i ) = REAL( i, working )
   END DO
   S%m = n ; S%n = n ; S%ne = n    ! S = diag(1:n)**2
!  CALL SMT_put( S%type, 'DIAGONAL', i )
   CALL SMT_put( S%type, 'COORDINATE', i )
!  ALLOCATE( S%val( n ) )
   ALLOCATE( S%row( n ), S%col( n ), S%val( n ) )
   DO i = 1, n
     S%row( i ) = i ; S%col( i ) = i
     S%val( i ) = REAL( i * i, working )
   END DO
   CALL LLSR_solve( m, n, power, weight, A, B, X, data, control, inform, S = S )
   IF ( inform%status == 0 ) THEN  !  Successful return
     DO l = 1, A%ne
       i =  A%row( l )
       B( i ) = B( i ) - A%val( l ) * X( A%col( l ) )
     END DO
     WRITE( 6, "( '   ||x||_S  recurred and calculated = ', 2ES16.8 )" )       &
!      inform%x_norm, SQRT( DOT_PRODUCT( X, X ) )
       inform%x_norm, SQRT( DOT_PRODUCT( X, S%val * X ) )
     WRITE( 6, "( ' ||Ax-b||_2 recurred and calculated = ', 2ES16.8 )" )       &
       inform%r_norm, SQRT( DOT_PRODUCT( B, B ) )
   ELSE                            !  Error returns
     WRITE( 6, "( ' LLSR_solve exit status = ', I0 ) " ) inform%status
   END IF
   CALL LLSR_terminate( data, control, inform ) ! delete workspace
   DEALLOCATE( A%row, A%col, A%val, S%val, A%type, S%type )
   END PROGRAM GALAHAD_LLSR2_EXAMPLE
