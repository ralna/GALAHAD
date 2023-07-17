   PROGRAM GALAHAD_LLSR_EXAMPLE  !  GALAHAD 4.1 - 2023-06-05 AT 13:15 GMT
   USE GALAHAD_LLSR_DOUBLE                         ! double precision version
   IMPLICIT NONE
   INTEGER, PARAMETER :: working = KIND( 1.0D+0 )  ! set precision
   REAL ( KIND = working ), PARAMETER :: one = 1.0_working, zero = 0.0_working
   INTEGER, PARAMETER :: n = 50, m = 2 * n         ! problem dimensions
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
   control%max_factorizations = 100
!  control%sbls_control%print_level = 1
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
   CALL LLSR_solve( m, n, power, weight, A, B, X, data, control, inform, S = S )
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
     WRITE( 6, "( ' LLSR_solve exit status = ', I0 ) " ) inform%status
   END IF
   CALL LLSR_terminate( data, control, inform ) ! delete workspace
   DEALLOCATE( A%row, A%col, A%val, S%val, A%type, S%type )
   END PROGRAM GALAHAD_LLSR_EXAMPLE
