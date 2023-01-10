! THIS VERSION: GALAHAD 4.1 - 2022-07-06 AT 11:30 GMT.
   PROGRAM GALAHAD_SLLS_SECOND_EXAMPLE ! reverse commmunication interface
   USE GALAHAD_SLLS_double             ! double precision version
   IMPLICIT NONE
   INTEGER, PARAMETER :: wp = KIND( 1.0D+0 ) ! set precision
   TYPE ( QPT_problem_type ) :: p
   TYPE ( SLLS_data_type ) :: data
   TYPE ( SLLS_control_type ) :: control
   TYPE ( SLLS_inform_type ) :: inform
   TYPE ( SLLS_reverse_type ) :: reverse
   TYPE ( GALAHAD_userdata_type ) :: userdata
   INTEGER, ALLOCATABLE, DIMENSION( : ) :: X_stat
   INTEGER :: i, j, k, l, nflag
   REAL ( KIND = wp ) :: val
   INTEGER, PARAMETER :: n = 3, m = 4, a_ne = 5
   INTEGER, ALLOCATABLE, DIMENSION( : ) :: A_row, A_ptr, FLAG
   REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: A_val
! start problem data
   ALLOCATE( p%B( m ), p%X( n ), X_stat( n ) )
   p%n = n ; p%m = m                          ! dimensions
   p%B = (/ 0.0_wp, 2.0_wp, 1.0_wp, 2.0_wp /) ! right-hand side
   p%X = 0.0_wp ! start from zero
!  sparse column storage format
   ALLOCATE( A_val( a_ne ), A_row( a_ne ), A_ptr( n + 1 ) )
   A_val = (/ 1.0_wp, 1.0_wp, 1.0_wp, 1.0_wp, 1.0_wp /) ! Jacobian A by columns
   A_row = (/ 1, 2, 2, 3, 4 /)                     ! row indices
   A_ptr = (/ 1, 3, 4, 6 /)                        ! pointers to column starts
! problem data complete
   CALL SLLS_initialize( data, control, inform ) ! Initialize control parameters
   control%print_level = 1                       ! print one line/iteration
   control%exact_arc_search = .FALSE.
   ALLOCATE( FLAG( n ) )
   nflag = 0 ; FLAG = 0  ! Flag if index already used in current (nflag) product
   inform%status = 1
10 CONTINUE ! Solve problem - reverse commmunication loop
     CALL SLLS_solve( p, X_stat, data, control, inform, userdata,              &
                      reverse = reverse )
     SELECT CASE ( inform%status )
     CASE ( 0 ) !  successful return
       WRITE( 6, "( /, ' SLLS: ', I0, ' iterations  ', /,                      &
      &     ' Optimal objective value =',                                      &
      &       ES12.4, /, ' Optimal solution = ', ( 5ES12.4 ) )" )              &
         inform%iter, inform%obj, p%X
     CASE ( 2 ) ! compute A * v
       reverse%P( : m ) = 0.0_wp
       DO j = 1, n
         val = reverse%V( j )
         DO k = A_ptr( j ), A_ptr( j + 1 ) - 1
           i = A_row( k )
           reverse%P( i ) = reverse%P( i ) + A_val( k ) * val
         END DO
       END DO
       GO TO 10
     CASE ( 3 ) ! compute A^T * v
       reverse%P( : n ) = 0.0_wp
       DO j = 1, n
         val = 0.0_wp
         DO k = A_ptr( j ), A_ptr( j + 1 ) - 1
           val = val + A_val( k ) * reverse%V( A_row( k ) )
         END DO
         reverse%P( j ) = val
       END DO
       GO TO 10
     CASE ( 4 ) ! compute A * sparse v
       reverse%P( : m ) = 0.0_wp
       DO l = reverse%nz_in_start, reverse%nz_in_end
         j = reverse%NZ_in( l )
         val = reverse%V( j )
         DO k = A_ptr( j ), A_ptr( j + 1 ) - 1
           i = A_row( k )
           reverse%P( i ) = reverse%P( i ) + A_val( k ) * val
         END DO
       END DO
       GO TO 10
     CASE ( 5 ) ! compute sparse( A * sparse v )
       nflag = nflag + 1
       reverse%nz_out_end = 0
       DO l = reverse%nz_in_start, reverse%nz_in_end
         j = reverse%NZ_in( l )
         val = reverse%V( j )
         DO k = A_ptr( j ), A_ptr( j + 1 ) - 1
           i = A_row( k )
           IF ( FLAG( i ) < nflag ) THEN
             FLAG( i ) = nflag
             reverse%P( i ) = A_val( k ) * val
             reverse%nz_out_end = reverse%nz_out_end + 1
             reverse%NZ_out( reverse%nz_out_end ) = i
           ELSE
             reverse%P( i ) = reverse%P( i ) + A_val( k ) * val
           END IF
         END DO
       END DO
       GO TO 10
     CASE ( 6 ) ! compute sparse( A^T * v )
       reverse%P( : n ) = 0.0_wp
       DO l = reverse%nz_in_start, reverse%nz_in_end
         j = reverse%NZ_in( l )
         val = 0.0_wp
         DO k = A_ptr( j ), A_ptr( j + 1 ) - 1
           val = val + A_val( k ) * reverse%V( A_row( k ) )
         END DO
         reverse%P( j ) = val
       END DO
       GO TO 10
     CASE DEFAULT ! error returns
       WRITE( 6, "( /, ' SLLS_solve exit status = ', I0 ) " ) inform%status
     END SELECT
   CALL SLLS_terminate( data, control, inform )  !  delete workspace
   DEALLOCATE( p%B, p%X, p%Z, X_stat, FLAG )
   DEALLOCATE( A_val, A_row, A_ptr )
   END PROGRAM GALAHAD_SLLS_SECOND_EXAMPLE
