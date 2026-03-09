! THIS VERSION: GALAHAD 5.5 - 2026-02-19 AT 10:20 GMT.
   PROGRAM GALAHAD_SLLS_THIRD_EXAMPLE ! subroutine evaluation interface
   USE GALAHAD_SLLS_double            ! double precision version
   IMPLICIT NONE
   INTEGER, PARAMETER :: wp = KIND( 1.0D+0 ) ! set precision
   TYPE ( QPT_problem_type ) :: p
   TYPE ( SLLS_data_type ) :: data
   TYPE ( SLLS_control_type ) :: control
   TYPE ( SLLS_inform_type ) :: inform
   TYPE ( USERDATA_type ) :: userdata
   INTEGER, PARAMETER :: n = 3, o = 4, m = 1, Ao_ne = 5
! partition userdata%integer so that it holds
!   o n  Ao_ptr      Ao_row
!  |1|2|3 to n+3|n+4 to n+3+a_ne|
! partition userdata%real so that it holds
!     Ao_val
!  |1 to Ao_ne|
   INTEGER, PARAMETER :: on = MAX( o, n )
   INTEGER, PARAMETER :: st_ptr = 3
   INTEGER, PARAMETER :: st_row = st_ptr + n + 1, st_val = 0
   INTEGER, PARAMETER :: len_integer = st_row + Ao_ne + 1, len_real = Ao_ne
   EXTERNAL :: APROD, ASCOL, AFPROD
! start problem data
   ALLOCATE( p%B( o ), p%X( n ), p%X_status( n ) )
   p%n = n ; p%o = o ; p%m = m  ! dimensions
   p%B = (/ 0.0_wp, 2.0_wp, 1.0_wp, 2.0_wp /) ! right-hand side
   p%X = 0.0_wp ! start from zero
!  sparse co-ordinate storage format
   ALLOCATE( userdata%integer( len_integer ), userdata%real( len_real ) )
   userdata%integer( 1 ) = o  ! load design matrix Ao data into userdata
   userdata%integer( 2 ) = n
   userdata%integer( st_ptr + 1 : st_ptr + n + 1 ) = (/ 1, 3, 4, 6 /)
   userdata%integer( st_row + 1 : st_row + Ao_ne ) = (/ 1, 2, 2, 3, 4 /)
   userdata%real( st_val + 1 : st_val + Ao_ne )                                &
     = (/ 1.0_wp, 1.0_wp, 1.0_wp, 1.0_wp, 1.0_wp /)
! problem data complete
   CALL SLLS_initialize( data, control, inform ) ! Initialize control parameters
!  control%print_level = 1                       ! print one line/iteration
   control%exact_arc_search = .FALSE.
!  load workspace into userdata
   inform%status = 1
   CALL SLLS_solve( p, data, control, inform, userdata, eval_APROD = APROD,    &
                    eval_ASCOL = ASCOL, eval_AFPROD = AFPROD )
   IF ( inform%status == 0 ) THEN             !  Successful return
     WRITE( 6, "( /, ' SLLS: ', I0, ' iterations  ', /,                        &
    &     ' Optimal objective value =',                                        &
    &       ES12.4, /, ' Optimal solution = ', ( 5ES12.4 ) )" )                &
       inform%iter, inform%obj, p%X
     WRITE( 6, "( ' Lagrange multiplier estimate =', ES12.4 )" ) p%Y
   ELSE                                       ! Error returns
     WRITE( 6, "( /, ' SLLS_solve exit status = ', I0 ) " ) inform%status
   END IF
   CALL SLLS_terminate( data, control, inform )  !  delete workspace
   DEALLOCATE( p%B, p%X, p%Y, p%Z, p%R, p%X_status )
   DEALLOCATE( userdata%integer, userdata%real )
   END PROGRAM GALAHAD_SLLS_THIRD_EXAMPLE

   SUBROUTINE APROD( status, userdata, transpose, V, P )
   USE GALAHAD_USERDATA_double
   INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
   INTEGER, INTENT( OUT ) :: status
   TYPE ( USERDATA_type ), INTENT( INOUT ) :: userdata
   LOGICAL, INTENT( IN ) :: transpose
   REAL ( KIND = wp ), DIMENSION( : ), INTENT( IN ) :: V
   REAL ( KIND = wp ), DIMENSION( : ), INTENT( INOUT ) :: P
   INTEGER :: i, j, k
   REAL ( KIND = wp ) :: val
!  recover problem data from userdata
   INTEGER :: o, n, st_ptr, st_row, st_val
   o = userdata%integer( 1 )
   n = userdata%integer( 2 )
   st_ptr = 3
   st_row = st_ptr + n + 1
   st_val = 0
   IF ( transpose ) THEN
     DO j = 1, n
       DO k = userdata%integer( st_ptr + j ),                                  &
              userdata%integer( st_ptr + j + 1 ) - 1
         P( j ) = P( j ) + userdata%real( st_val + k ) *                       &
                     V( userdata%integer( st_row + k ) )
       END DO
     END DO
   ELSE
     DO j = 1, n
       val = V( j )
       DO k = userdata%integer( st_ptr + j ),                                  &
              userdata%integer( st_ptr + j + 1 ) - 1
         i = userdata%integer( st_row + k )
         P( i ) = P( i ) + userdata%real( st_val + k ) * val
       END DO
     END DO
   END IF
   status = 0
   RETURN
   END SUBROUTINE APROD

   SUBROUTINE ASCOL( status, userdata, index, P, IP, lp )
   USE GALAHAD_USERDATA_double
   INTEGER ( KIND = ip_ ), INTENT( OUT ) :: status
   TYPE ( USERDATA_type ), INTENT( INOUT ) :: userdata
   INTEGER ( KIND = ip_ ), INTENT( IN ) :: index
   REAL ( KIND = rp_ ), DIMENSION( : ), INTENT( OUT ) :: P
   INTEGER ( KIND = ip_ ), DIMENSION( : ), INTENT( INOUT ) :: IP
   INTEGER ( KIND = ip_ ), INTENT( INOUT ) :: lp
   INTEGER ( KIND = ip_ ) :: k
!  recover problem data from userdata
   INTEGER ( KIND = ip_ ) :: o, n, st_ptr, st_row, st_val
   o = userdata%integer( 1 )
   n = userdata%integer( 2 )
   st_ptr = 3
   st_row = st_ptr + n + 1
   st_val = 0
   lp = 0
   DO k = userdata%integer( st_ptr + index ),                                  &
          userdata%integer( st_ptr + index + 1 ) - 1
     lp = lp + 1
     P( lp ) = userdata%real( st_val + k )
     IP( lp ) = userdata%integer( st_row + k )
   END DO
   status = 0
   RETURN
   END SUBROUTINE ASCOL

   SUBROUTINE AFPROD( status, userdata, transpose, V, P, FREE, n_free )
   USE GALAHAD_USERDATA_double
   INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
   INTEGER, INTENT( OUT ) :: status
   TYPE ( USERDATA_type ), INTENT( INOUT ) :: userdata
   LOGICAL, INTENT( IN ) :: transpose
   INTEGER, INTENT( IN ) :: n_free
   INTEGER, INTENT( IN ), DIMENSION( : ) :: FREE
   REAL ( KIND = wp ), DIMENSION( : ), INTENT( IN ) :: V
   REAL ( KIND = wp ), DIMENSION( : ), INTENT( OUT ) :: P
   INTEGER :: i, j, k, l
   REAL ( KIND = wp ) :: val
!  recover problem data from userdata
   INTEGER :: o, n, st_ptr, st_row, st_val
   o = userdata%integer( 1 )
   n = userdata%integer( 2 )
   st_ptr = 3
   st_row = st_ptr + n + 1
   st_val = 0
   IF ( transpose ) THEN
     DO l = 1, n_free
       j = FREE( l )
       val = 0.0_wp
       DO k = userdata%integer( st_ptr + j ),                                  &
              userdata%integer( st_ptr + j + 1 ) - 1
         val = val + userdata%real( st_val + k ) *                             &
                 V( userdata%integer( st_row + k ) )
       END DO
       P( j ) = val
     END DO
   ELSE
     P( : o ) = 0.0_wp
     DO l = 1, n_free
       j = FREE( l )
       val = V( j )
       DO k = userdata%integer( st_ptr + j ),                                  &
              userdata%integer( st_ptr + j + 1 ) - 1
         i = userdata%integer( st_row + k )
         P( i ) = P( i ) + userdata%real( st_val + k ) * val
       END DO
     END DO
   END IF
   status = 0
   RETURN
   END SUBROUTINE AFPROD
