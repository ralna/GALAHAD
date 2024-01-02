! THIS VERSION: GALAHAD 4.3 - 2023-12-29 AT 13:50 GMT.
   PROGRAM GALAHAD_BLLS_SECOND_EXAMPLE ! reverse commmunication interface
   USE GALAHAD_BLLS_double             ! double precision version
   IMPLICIT NONE
   INTEGER, PARAMETER :: wp = KIND( 1.0D+0 ) ! set precision
   REAL ( KIND = wp ), PARAMETER :: infinity = 10.0_wp ** 20
   TYPE ( QPT_problem_type ) :: p
   TYPE ( BLLS_data_type ) :: data
   TYPE ( BLLS_control_type ) :: control
   TYPE ( BLLS_inform_type ) :: inform
   TYPE ( BLLS_reverse_type ) :: reverse
   TYPE ( GALAHAD_userdata_type ) :: userdata
   INTEGER, ALLOCATABLE, DIMENSION( : ) :: X_stat
   INTEGER :: i, j, k, l, nflag
   REAL ( KIND = wp ) :: val
   INTEGER, PARAMETER :: n = 3, o = 4, ao_ne = 5
   INTEGER, ALLOCATABLE, DIMENSION( : ) :: Ao_row, Ao_ptr, FLAG
   REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: Ao_val
! start problem data
   ALLOCATE( p%B( o ), p%X_l( n ), p%X_u( n ), p%X( n ), X_stat( n ) )
   p%n = n ; p%o = o                          ! dimensions
   p%B = (/ 0.0_wp, 2.0_wp, 1.0_wp, 2.0_wp /) ! right-hand side
   p%X_l = (/ - 1.0_wp, - infinity, 0.0_wp /) ! variable lower bound
   p%X_u = (/ infinity, 1.0_wp, 2.0_wp /)     ! variable upper bound
   p%X = 0.0_wp ! start from zero
!  sparse column storage format
   ALLOCATE( Ao_val( ao_ne ), Ao_row( ao_ne ), Ao_ptr( n + 1 ) )
   Ao_val = (/ 1.0_wp, 1.0_wp, 1.0_wp, 1.0_wp, 1.0_wp /) ! Jacobian A by columns
   Ao_row = (/ 1, 2, 2, 3, 4 /)                     ! row indices
   Ao_ptr = (/ 1, 3, 4, 6 /)                        ! pointers to column starts
! problem data complete
   CALL BLLS_initialize( data, control, inform ) ! Initialize control parameters
   control%infinity = infinity                   ! Set infinity
   control%print_level = 1                       ! print one line/iteration
   control%exact_arc_search = .FALSE.
   ALLOCATE( FLAG( n ) )
   nflag = 0 ; FLAG = 0  ! Flag if index already used in current (nflag) product
   inform%status = 1
10 CONTINUE ! Solve problem - reverse commmunication loop
     CALL BLLS_solve( p, X_stat, data, control, inform, userdata,              &
                      reverse = reverse )

     SELECT CASE ( inform%status )
     CASE ( 0 ) !  successful return
       WRITE( 6, "( /, ' BLLS: ', I0, ' iterations  ', /,                      &
      &     ' Optimal objective value =',                                      &
      &       ES12.4, /, ' Optimal solution = ', ( 5ES12.4 ) )" )              &
         inform%iter, inform%obj, p%X
     CASE ( 2 ) ! compute A * v
       reverse%P( : o ) = 0.0_wp
       DO j = 1, n
         val = reverse%V( j )
         DO k = Ao_ptr( j ), Ao_ptr( j + 1 ) - 1
           i = Ao_row( k )
           reverse%P( i ) = reverse%P( i ) + Ao_val( k ) * val
         END DO
       END DO
       GO TO 10
     CASE ( 3 ) ! compute A^T * v
       reverse%P( : n ) = 0.0_wp
       DO j = 1, n
         val = 0.0_wp
         DO k = Ao_ptr( j ), Ao_ptr( j + 1 ) - 1
           val = val + Ao_val( k ) * reverse%V( Ao_row( k ) )
         END DO
         reverse%P( j ) = val
       END DO
       GO TO 10
     CASE ( 4 ) ! compute A * sparse v
       reverse%P( : o ) = 0.0_wp
       DO l = reverse%nz_in_start, reverse%nz_in_end
         j = reverse%NZ_in( l )
         val = reverse%V( j )
         DO k = Ao_ptr( j ), Ao_ptr( j + 1 ) - 1
           i = Ao_row( k )
           reverse%P( i ) = reverse%P( i ) + Ao_val( k ) * val
         END DO
       END DO
       GO TO 10
     CASE ( 5 ) ! compute sparse( A * sparse v )
       nflag = nflag + 1
       reverse%nz_out_end = 0
       DO l = reverse%nz_in_start, reverse%nz_in_end
         j = reverse%NZ_in( l )
         val = reverse%V( j )
         DO k = Ao_ptr( j ), Ao_ptr( j + 1 ) - 1
           i = Ao_row( k )
           IF ( FLAG( i ) < nflag ) THEN
             FLAG( i ) = nflag
             reverse%P( i ) = Ao_val( k ) * val
             reverse%nz_out_end = reverse%nz_out_end + 1
             reverse%NZ_out( reverse%nz_out_end ) = i
           ELSE
             reverse%P( i ) = reverse%P( i ) + Ao_val( k ) * val
           END IF
         END DO
       END DO
       GO TO 10
     CASE ( 6 ) ! compute sparse( A^T * v )
       reverse%P( : n ) = 0.0_wp
       DO l = reverse%nz_in_start, reverse%nz_in_end
         j = reverse%NZ_in( l )
         val = 0.0_wp
         DO k = Ao_ptr( j ), Ao_ptr( j + 1 ) - 1
           val = val + Ao_val( k ) * reverse%V( Ao_row( k ) )
         END DO
         reverse%P( j ) = val
       END DO
       GO TO 10
     CASE DEFAULT ! error returns
       WRITE( 6, "( /, ' BLLS_solve exit status = ', I0 ) " ) inform%status
     END SELECT
   CALL BLLS_terminate( data, control, inform )  !  delete workspace
   DEALLOCATE( p%B, p%X, p%X_l, p%X_u, p%Z, p%R, p%G, X_stat, FLAG )
   DEALLOCATE( Ao_val, Ao_row, Ao_ptr )
   END PROGRAM GALAHAD_BLLS_SECOND_EXAMPLE
