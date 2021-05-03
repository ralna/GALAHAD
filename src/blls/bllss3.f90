! THIS VERSION: GALAHAD 3.3 - 07/04/2021 AT 06:50 GMT.
   PROGRAM GALAHAD_BLLS_THIRD_EXAMPLE ! subroutine evaluation interface
   USE GALAHAD_BLLS_double            ! double precision version
   IMPLICIT NONE
   INTEGER, PARAMETER :: wp = KIND( 1.0D+0 ) ! set precision
   REAL ( KIND = wp ), PARAMETER :: infinity = 10.0_wp ** 20
   TYPE ( QPT_problem_type ) :: p
   TYPE ( BLLS_data_type ) :: data
   TYPE ( BLLS_control_type ) :: control
   TYPE ( BLLS_inform_type ) :: inform
   TYPE ( GALAHAD_userdata_type ) :: userdata
   INTEGER, ALLOCATABLE, DIMENSION( : ) :: X_stat
   INTEGER, PARAMETER :: n = 3, m = 4, a_ne = 5
! partition userdata%integer so that it holds
!   m n nflag  flag       a_ptr          a_row
!  |1|2|  3  |4 to n+3 |n+4 to 2n+4|2n+5 to 2n+4+a_ne|
! partition userdata%real so that it holds
!     a_val
!  |1 to a_ne|
   INTEGER, PARAMETER :: mn = MAX( m, n )
   INTEGER, PARAMETER :: nflag = 3, st_flag = 3, st_ptr = st_flag + mn
   INTEGER, PARAMETER :: st_row = st_ptr + n + 1, st_val = 0
   INTEGER, PARAMETER :: len_integer = st_row + a_ne + 1, len_real = a_ne
   EXTERNAL :: APROD, ASPROD, AFPROD
! start problem data
   ALLOCATE( p%B( m ), p%X_l( n ), p%X_u( n ), p%X( n ), X_stat( n ) )
   p%n = n ; p%m = m                          ! dimensions
   p%B = (/ 0.0_wp, 2.0_wp, 1.0_wp, 2.0_wp /) ! right-hand side
   p%X_l = (/ - 1.0_wp, - infinity, 0.0_wp /) ! variable lower bound
   p%X_u = (/ infinity, 1.0_wp, 2.0_wp /)     ! variable upper bound
   p%X = 0.0_wp ! start from zero
!  sparse co-ordinate storage format
   ALLOCATE( userdata%integer( len_integer ), userdata%real( len_real ) )
   userdata%integer( 1 ) = m                  ! load Jacobian data into userdata
   userdata%integer( 2 ) = n
   userdata%integer( st_ptr + 1 : st_ptr + n + 1 ) = (/ 1, 3, 4, 6 /)
   userdata%integer( st_row + 1 : st_row + a_ne ) = (/ 1, 2, 2, 3, 4 /)
   userdata%real( st_val + 1 : st_val + a_ne )                                 &
     = (/ 1.0_wp, 1.0_wp, 1.0_wp, 1.0_wp, 1.0_wp /)
! problem data complete
   CALL BLLS_initialize( data, control, inform ) ! Initialize control parameters
   control%infinity = infinity                   ! Set infinity
   control%print_level = 1                       ! print one line/iteration
   control%exact_arc_search = .FALSE.
!  load workspace into userdata
   userdata%integer( nflag ) = 0
   userdata%integer( st_flag + 1 : st_flag + mn ) = 0
   inform%status = 1
   CALL BLLS_solve( p, X_stat, data, control, inform, userdata,                &
                    eval_APROD = APROD, eval_ASPROD = ASPROD,                  &
                    eval_AFPROD = AFPROD )
   IF ( inform%status == 0 ) THEN             !  Successful return
     WRITE( 6, "( /, ' BLLS: ', I0, ' iterations  ', /,                        &
    &     ' Optimal objective value =',                                        &
    &       ES12.4, /, ' Optimal solution = ', ( 5ES12.4 ) )" )                &
     inform%iter, inform%obj, p%X
   ELSE                                       ! Error returns
     WRITE( 6, "( /, ' BLLS_solve exit status = ', I0 ) " ) inform%status
   END IF
   CALL BLLS_terminate( data, control, inform )  !  delete workspace
   DEALLOCATE( p%B, p%X, p%X_l, p%X_u, p%Z, X_stat )
   DEALLOCATE( userdata%integer, userdata%real )
   END PROGRAM GALAHAD_BLLS_THIRD_EXAMPLE

   SUBROUTINE APROD( status, userdata, transpose, V, P )
   USE GALAHAD_USERDATA_double
   INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
   INTEGER, INTENT( OUT ) :: status
   TYPE ( GALAHAD_userdata_type ), INTENT( INOUT ) :: userdata
   LOGICAL, INTENT( IN ) :: transpose
   REAL ( KIND = wp ), DIMENSION( : ), INTENT( IN ) :: V
   REAL ( KIND = wp ), DIMENSION( : ), INTENT( INOUT ) :: P
   INTEGER :: i, j, k
   REAL ( KIND = wp ) :: val
!  recover problem data from userdata
   INTEGER :: m, n, nflag, st_flag, st_ptr, st_row, st_val
   m = userdata%integer( 1 )
   n = userdata%integer( 2 )
   nflag = 3
   st_flag = 3
   st_ptr = st_flag + MAX( m, n )
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

   SUBROUTINE ASPROD( status, userdata, V, P, NZ_in, nz_in_start, nz_in_end,   &
                      NZ_out, nz_out_end )
   USE GALAHAD_USERDATA_double
   INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
   INTEGER, INTENT( OUT ) :: status
   TYPE ( GALAHAD_userdata_type ), INTENT( INOUT ) :: userdata
   REAL ( KIND = wp ), DIMENSION( : ), INTENT( IN ) :: V
   REAL ( KIND = wp ), DIMENSION( : ), INTENT( OUT ) :: P
   INTEGER, OPTIONAL, INTENT( IN ) :: nz_in_start, nz_in_end
   INTEGER, OPTIONAL, INTENT( INOUT ) :: nz_out_end
   INTEGER, DIMENSION( : ), OPTIONAL, INTENT( IN ) :: NZ_in
   INTEGER, DIMENSION( : ), OPTIONAL, INTENT( INOUT ) :: NZ_out
   INTEGER :: i, j, k, l
   REAL ( KIND = wp ) :: val
!  recover problem data from userdata
   INTEGER :: m, n, nflag, st_flag, st_ptr, st_row, st_val
   IF ( PRESENT( NZ_in ) ) THEN
     IF ( .NOT. ( PRESENT( nz_in_start ) .AND. PRESENT( nz_in_end ) ) ) THEN
         status = - 1 ; RETURN
     END IF
   END IF
   m = userdata%integer( 1 )
   n = userdata%integer( 2 )
   nflag = 3
   st_flag = 3
   st_ptr = st_flag + MAX( m, n )
   st_row = st_ptr + n + 1
   st_val = 0
   IF ( PRESENT( NZ_in ) ) THEN
     IF ( PRESENT( NZ_out ) ) THEN
       IF ( .NOT. PRESENT( nz_out_end ) ) THEN
         status = - 1 ; RETURN
       END IF
       userdata%integer( nflag ) = userdata%integer( nflag ) + 1
       nz_out_end = 0
       DO l = nz_in_start, nz_in_end
         j = NZ_in( l )
         val = V( j )
         DO k = userdata%integer( st_ptr + j ),                                &
                userdata%integer( st_ptr + j + 1 ) - 1
           i = userdata%integer( st_row + k )
           IF ( userdata%integer( st_flag + i ) <                              &
                userdata%integer( nflag )  ) THEN
             userdata%integer( st_flag + i ) = userdata%integer( nflag )
             P( i ) = userdata%real( st_val + k ) * val
             nz_out_end = nz_out_end + 1
             NZ_out( nz_out_end ) = i
           ELSE
             P( i ) = P( i ) + userdata%real( st_val + k ) * val
           END IF
         END DO
       END DO
     ELSE
       P( : m ) = 0.0_wp
       DO l = nz_in_start, nz_in_end
         j = NZ_in( l )
         val = V( j )
         DO k = userdata%integer( st_ptr + j ),                                &
                userdata%integer( st_ptr + j + 1 ) - 1
           i = userdata%integer( st_row + k )
           P( i ) = P( i ) + userdata%real( st_val + k ) * val
         END DO
       END DO
     END IF
   ELSE
     IF ( PRESENT( NZ_out ) ) THEN
       IF ( .NOT. PRESENT( nz_out_end ) ) THEN
         status = - 1 ; RETURN
       END IF
       userdata%integer( nflag ) = userdata%integer( nflag ) + 1
       nz_out_end = 0
       DO j = 1, n
         val = V( j )
         DO k = userdata%integer( st_ptr + j ),                                &
                userdata%integer( st_ptr + j + 1 ) - 1
           i = userdata%integer( st_row + k )
           IF ( userdata%integer( st_flag + i ) <                              &
                userdata%integer( nflag )  ) THEN
             userdata%integer( st_flag + i ) = userdata%integer( nflag )
             P( i ) = userdata%real( st_val + k ) * val
             nz_out_end = nz_out_end + 1
             NZ_out( nz_out_end ) = i
           ELSE
             P( i ) = P( i ) + userdata%real( st_val + k ) * val
           END IF
         END DO
       END DO
     ELSE
       P( : m ) = 0.0_wp
       DO j = 1, n
         val = V( j )
         DO k = userdata%integer( st_ptr + j ),                                &
                userdata%integer( st_ptr + j + 1 ) - 1
           i = userdata%integer( st_row + k )
           P( i ) = P( i ) + userdata%real( st_val + k ) * val
         END DO
       END DO
     END IF
   END IF
   status = 0
   RETURN
   END SUBROUTINE ASPROD

   SUBROUTINE AFPROD( status, userdata, transpose, V, P, FREE, n_free )
   USE GALAHAD_USERDATA_double
   INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
   INTEGER, INTENT( OUT ) :: status
   TYPE ( GALAHAD_userdata_type ), INTENT( INOUT ) :: userdata
   LOGICAL, INTENT( IN ) :: transpose
   INTEGER, INTENT( IN ) :: n_free
   INTEGER, INTENT( IN ), DIMENSION( : ) :: FREE
   REAL ( KIND = wp ), DIMENSION( : ), INTENT( IN ) :: V
   REAL ( KIND = wp ), DIMENSION( : ), INTENT( OUT ) :: P
   INTEGER :: i, j, k, l
   REAL ( KIND = wp ) :: val
!  recover problem data from userdata
   INTEGER :: m, n, nflag, st_flag, st_ptr, st_row, st_val
   m = userdata%integer( 1 )
   n = userdata%integer( 2 )
   nflag = 3
   st_flag = 3
   st_ptr = st_flag + MAX( m, n )
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
     P( : m ) = 0.0_wp
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
