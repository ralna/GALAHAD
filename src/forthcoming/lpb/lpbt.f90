! THIS VERSION: GALAHAD 2.1 - 22/03/2007 AT 09:00 GMT.
   PROGRAM GALAHAD_LPB_EXAMPLE
   USE GALAHAD_LPB_double                            ! double precision version
   IMPLICIT NONE
   INTEGER, PARAMETER :: wp = KIND( 1.0D+0 ) ! set precision
   REAL ( KIND = wp ), PARAMETER :: infty = 10.0_wp ** 20
   TYPE ( QPT_problem_type ) :: p
   TYPE ( LPB_data_type ) :: data
   TYPE ( LPB_control_type ) :: control        
   TYPE ( LPB_inform_type ) :: info
   INTEGER :: n, m, h_ne, a_ne, smt_stat
   INTEGER :: data_storage_type, i, status, scratch_out = 56
   CHARACTER ( len = 1 ) :: st

   n = 3 ; m = 2 ; h_ne = 4 ; a_ne = 4 
   ALLOCATE( p%G( n ), p%X_l( n ), p%X_u( n ) )
   ALLOCATE( p%C( m ), p%C_l( m ), p%C_u( m ) )
   ALLOCATE( p%X( n ), p%Y( m ), p%Z( n ) )

!  ================
!  error exit tests
!  ================

   WRITE( 6, "( /, ' error exit tests ', / )" )
   p%new_problem_structure = .TRUE.
   p%n = n ; p%m = m ; p%f = 1.0_wp
   p%G = (/ 0.0_wp, 2.0_wp, 0.0_wp /)
   p%C_l = (/ 1.0_wp, 2.0_wp /)
   p%C_u = (/ 4.0_wp, infty /)
   p%X_l = (/ - 1.0_wp, - infty, - infty /)
   p%X_u = (/ 1.0_wp, infty, 2.0_wp /)

   CALL LPB_initialize( data, control )
   control%infinity = infty
   control%restore_problem = 1

!  tests for status = - 1 ... - 10

   DO status = 1, 10
     IF ( status == 2 ) CYCLE
     IF ( status == 3 ) CYCLE
     IF ( status == 7 ) CYCLE
     IF ( status == 8 ) CYCLE
     ALLOCATE( p%A%val( a_ne ), p%A%row( 0 ), p%A%col( a_ne ) )
     IF ( ALLOCATED( p%A%type ) ) DEALLOCATE( p%A%type )
     CALL SMT_put( p%A%type, 'SPARSE_BY_ROWS', smt_stat ) 
     p%A%val = (/ 2.0_wp, 1.0_wp, 1.0_wp, 1.0_wp /)
     p%A%col = (/ 1, 2, 2, 3 /)
     p%A%ptr = (/ 1, 3, 5 /)
     p%X = 0.0_wp ; p%Y = 0.0_wp ; p%Z = 0.0_wp

     IF ( status == 1 ) THEN
       p%n = 0 ; p%m = - 1
     ELSE IF ( status == 4 ) THEN 

     ELSE IF ( status == 5 ) THEN 
       p%X_u( 1 ) = - 2.0_wp
     ELSE IF ( status == 6 ) THEN
!      control%print_level = 1
       p%new_problem_structure = .TRUE.
       p%n = n ; p%m = m ; p%f = 1.0_wp
       p%G = (/ 0.0_wp, 2.0_wp, 0.0_wp /)
       p%C_l = (/ 1.0_wp, 2.0_wp /)
       p%C_u = (/ 4.0_wp, infty /)
       p%X_l = (/ - 1.0_wp, 8.0_wp, - infty /)
       p%X_u = (/ 1.0_wp, infty, 2.0_wp /)
     ELSE IF ( status == 9 ) THEN
       control%print_level = 0
!      control%print_level = 1
       p%new_problem_structure = .TRUE.
       p%n = n ; p%m = m ; p%f = 1.0_wp
       p%G = (/ 0.0_wp, 2.0_wp, 0.0_wp /)
       p%C_l = (/ 1.0_wp, 2.0_wp /)
       p%C_u = (/ 4.0_wp, infty /)
       p%X_l = (/ - 1.0_wp, - infty, - infty /)
       p%X_u = (/ 1.0_wp, infty, 2.0_wp /)
       control%initial_radius  = EPSILON( 1.0_wp ) ** 2
     ELSE IF ( status == 10 ) THEN
       control%maxit = 1
!      control%print_level = 1
     ELSE
     END IF

     CALL LPB_solve( p, data, control, info )
     IF ( info%status == 0 ) THEN
       WRITE( 6, "( I2, ':', I6, ' iterations. Optimal objective value = ',    &
     &      F6.1, ' status = ', I6 )" ) status, info%iter, info%obj, info%status
     ELSE
       WRITE( 6, "(I2, ': LPB_solve exit status = ', I6 )") status, info%status
     END IF
     DEALLOCATE( p%A%val, p%A%row, p%A%col )
     DEALLOCATE( p%A%type )
     IF ( status == 1 ) THEN
       p%n = n ; p%m = m
     ELSE IF ( status == 2 ) THEN
     ELSE IF ( status == 3 ) THEN
     ELSE IF ( status == 5 ) THEN
       p%X_u( 1 ) = 1.0_wp
     ELSE IF ( status == 8 ) THEN
       control%stop_p  = EPSILON( 1.0_wp ) ** 0.33
       control%stop_c  = EPSILON( 1.0_wp ) ** 0.33
       control%stop_d  = EPSILON( 1.0_wp ) ** 0.33
     ELSE IF ( status == 9 ) THEN
       control%initial_radius = - 1.0_wp
     ELSE
     END IF
   END DO

   CALL LPB_terminate( data, control, info )
   DEALLOCATE( p%G, p%X_l, p%X_u, p%C_l, p%C_u )
   DEALLOCATE( p%X, p%Y, p%Z, p%C )
   DEALLOCATE( p%A%ptr )

!  special test for status = - 11

   status = 11
   n = 1 ; m = 0 ; h_ne = 1 ; a_ne = 0
   IF ( ALLOCATED( p%A%type ) ) DEALLOCATE( p%A%type )
   CALL SMT_put( p%A%type, 'COORDINATE', smt_stat )
   ALLOCATE( p%G( n ), p%X_l( n ), p%X_u( n ) )
   ALLOCATE( p%C( m ), p%C_l( m ), p%C_u( m ) )
   ALLOCATE( p%X( n ), p%Y( m ), p%Z( n ) )
   ALLOCATE( p%A%ptr( m + 1 ) )
   ALLOCATE( p%A%val( a_ne ), p%A%row( a_ne ), p%A%col( a_ne ) )
   p%new_problem_structure = .TRUE.
   p%n = n ; p%m = m ; p%A%ne = a_ne 
   p%f = 0.0_wp
   p%G = (/ 0.0_wp /)
   p%X_l = (/ 0.0_wp /)
   p%X_u = (/ infty /)
   CALL LPB_initialize( data, control )
   control%infinity = infty
   control%restore_problem = 1
!  control%print_level = 1
   p%X = 0.0_wp ; p%Y = 0.0_wp ; p%Z = 0.0_wp
   CALL LPB_solve( p, data, control, info )
   IF ( info%status == 0 ) THEN
       WRITE( 6, "( I2, ':', I6, ' iterations. Optimal objective value = ',    &
     &      F6.1, ' status = ', I6 )" ) status, info%iter, info%obj, info%status
      write(6,*) info%obj
     ELSE
       WRITE( 6, "(I2, ': LPB_solve exit status = ', I6 )") status, info%status
   END IF
   CALL LPB_terminate( data, control, info )
   DEALLOCATE( p%A%val, p%A%row, p%A%col )
   DEALLOCATE( p%G, p%X_l, p%X_u, p%C_l, p%C_u )
   DEALLOCATE( p%X, p%Y, p%Z, p%C )
   DEALLOCATE( p%A%type )

!  =====================================
!  basic test of various storage formats
!  =====================================

   WRITE( 6, "( /, ' basic tests of storage formats ', / )" )

   n = 3 ; m = 2 ; h_ne = 4 ; a_ne = 4 
   ALLOCATE( p%G( n ), p%X_l( n ), p%X_u( n ) )
   ALLOCATE( p%C( m ), p%C_l( m ), p%C_u( m ) )
   ALLOCATE( p%X( n ), p%Y( m ), p%Z( n ) )
   ALLOCATE( p%A%ptr( m + 1 ) )

   p%n = n ; p%m = m ; p%f = 0.96_wp
   p%G = (/ 0.0_wp, 2.0_wp, 0.0_wp /)
   p%C_l = (/ 1.0_wp, 2.0_wp /)
   p%C_u = (/ 4.0_wp, infty /)
   p%X_l = (/ - 1.0_wp, - infty, - infty /)
   p%X_u = (/ 1.0_wp, infty, 2.0_wp /)

   DO data_storage_type = -3, 0
     CALL LPB_initialize( data, control )
     control%infinity = infty
     control%restore_problem = 2
     p%new_problem_structure = .TRUE.
     IF ( data_storage_type == 0 ) THEN           ! sparse co-ordinate storage
       st = 'C'
       ALLOCATE( p%A%val( a_ne ), p%A%row( a_ne ), p%A%col( a_ne ) )
       IF ( ALLOCATED( p%A%type ) ) DEALLOCATE( p%A%type )
       CALL SMT_put( p%A%type, 'COORDINATE', smt_stat )
       p%A%row = (/ 1, 1, 2, 2 /)
       p%A%col = (/ 1, 2, 2, 3 /) ; p%A%ne = a_ne
     ELSE IF ( data_storage_type == - 1 ) THEN     ! sparse row-wise storage
       st = 'R'
       ALLOCATE( p%A%val( a_ne ), p%A%row( 0 ), p%A%col( a_ne ) )
       IF ( ALLOCATED( p%A%type ) ) DEALLOCATE( p%A%type )
       CALL SMT_put( p%A%type, 'SPARSE_BY_ROWS', smt_stat )
       p%A%col = (/ 1, 2, 2, 3 /)
       p%A%ptr = (/ 1, 3, 5 /)
     ELSE IF ( data_storage_type == - 2 ) THEN      ! dense storage
       st = 'D'
       ALLOCATE( p%A%val(n*m), p%A%row(0), p%A%col(n*m) )
       IF ( ALLOCATED( p%A%type ) ) DEALLOCATE( p%A%type )
       CALL SMT_put( p%A%type, 'DENSE', smt_stat )
     ELSE IF ( data_storage_type == - 3 ) THEN      ! diagonal H, dense A
       st = 'I'
       ALLOCATE( p%A%val(n*m), p%A%row(0), p%A%col(n*m) )
       IF ( ALLOCATED( p%A%type ) ) DEALLOCATE( p%A%type )
       CALL SMT_put( p%A%type, 'DENSE', smt_stat )
     END IF

!  test with new and existing data

     DO i = 1, 2
       IF ( data_storage_type == 0 ) THEN          ! sparse co-ordinate storage
         p%A%val = (/ 2.0_wp, 1.0_wp, 1.0_wp, 1.0_wp /)
       ELSE IF ( data_storage_type == - 1 ) THEN    !  sparse row-wise storage
         p%A%val = (/ 2.0_wp, 1.0_wp, 1.0_wp, 1.0_wp /)
       ELSE IF ( data_storage_type == - 2 ) THEN    !  dense storage
         p%A%val = (/ 2.0_wp, 1.0_wp, 0.0_wp, 0.0_wp, 1.0_wp, 1.0_wp /)
       ELSE IF ( data_storage_type == - 3 ) THEN    !  diagonal/dense storage
         p%A%val = (/ 2.0_wp, 1.0_wp, 0.0_wp, 0.0_wp, 1.0_wp, 1.0_wp /)
       END IF
       p%X = 0.0_wp ; p%Y = 0.0_wp ; p%Z = 0.0_wp
       CALL LPB_solve( p, data, control, info )
      IF ( info%status == 0 ) THEN
         WRITE( 6, "( A1,I1,':', I6, ' iterations. Optimal objective value = ',&
       &       F6.1, ' status = ', I6 )") st, i,info%iter, info%obj, info%status
       ELSE
         WRITE( 6, "( A1, I1,': LPB_solve exit status = ', I6 ) " )           &
           st, i, info%status
       END IF
     END DO
     CALL LPB_terminate( data, control, info )
     DEALLOCATE( p%A%val, p%A%row, p%A%col )
     DEALLOCATE( p%A%type )
!    STOP
   END DO
   DEALLOCATE( p%G, p%X_l, p%X_u, p%C_l, p%C_u )
   DEALLOCATE( p%X, p%Y, p%Z, p%C )
   DEALLOCATE( p%A%ptr )

!  =============================
!  basic test of various options
!  =============================

   WRITE( 6, "( /, ' basic tests of options ', / )" )

   n = 2 ; m = 1 ; h_ne = 2 ; a_ne = 2 
   ALLOCATE( p%G( n ), p%X_l( n ), p%X_u( n ) )
   ALLOCATE( p%C( m ), p%C_l( m ), p%C_u( m ) )
   ALLOCATE( p%X( n ), p%Y( m ), p%Z( n ) )
   ALLOCATE( p%A%ptr( m + 1 ) )

   p%n = n ; p%m = m ; p%f = 0.05_wp
   p%G = (/ 0.0_wp, 0.0_wp /)
   p%C_l = (/ 1.0_wp /)
   p%C_u = (/ 1.0_wp /)
   p%X_l = (/ 0.0_wp, 0.0_wp /)
   p%X_u = (/ 2.0_wp, 3.0_wp /)

   p%new_problem_structure = .TRUE.
   ALLOCATE( p%A%val( a_ne ), p%A%row( 0 ), p%A%col( a_ne ) )
   IF ( ALLOCATED( p%A%type ) ) DEALLOCATE( p%A%type )
   CALL SMT_put( p%A%type, 'SPARSE_BY_ROWS', smt_stat )
   p%A%col = (/ 1, 2 /)
   p%A%ptr = (/ 1, 3 /)
   CALL LPB_initialize( data, control )
   control%infinity = infty
   control%restore_problem = 2
   
!  test with new and existing data

   DO i = 0, 15
     IF ( i == 0 ) THEN
       control%precon = 0
     ELSE IF ( i == 1 ) THEN
       control%precon = 1
     ELSE IF ( i == 2 ) THEN
       control%precon = 2
     ELSE IF ( i == 3 ) THEN
       control%precon = 3
     ELSE IF ( i == 4 ) THEN
       control%precon = 5
     ELSE IF ( i == 5 ) THEN     
       control%factor = - 1
     ELSE IF ( i == 6 ) THEN     
       control%factor = 1
     ELSE IF ( i == 7 ) THEN     
       control%max_col = 0
     ELSE IF ( i == 8 ) THEN     
       control%factor = 2
       control%precon = 0
     ELSE IF ( i == 9 ) THEN
!      control%print_level = 2
       control%precon = 1
     ELSE IF ( i == 10 ) THEN
       control%precon = 2
     ELSE IF ( i == 11 ) THEN
       control%precon = 3
     ELSE IF ( i == 12 ) THEN
       control%precon = 5
     ELSE IF ( i == 13 ) THEN
       control%center = .FALSE.
     ELSE IF ( i == 14 ) THEN
       control%primal = .TRUE.       
     ELSE IF ( i == 15 ) THEN
       control%feasol = .FALSE.
     END IF

     p%A%val = (/ 1.0_wp, 1.0_wp /)
     p%X = 0.0_wp ; p%Y = 0.0_wp ; p%Z = 0.0_wp
!    control%print_level = 4
     CALL LPB_solve( p, data, control, info )
!    write(6,"('x=', 2ES12.4)") p%X
     IF ( info%status == 0 ) THEN
       WRITE( 6, "( I2, ':', I6, ' iterations. Optimal objective value = ',    &
     &       F6.1, ' status = ', I6 )" ) i, info%iter, info%obj, info%status
     ELSE
       WRITE( 6, "( I2, ': LPB_solve exit status = ', I6 ) " ) i, info%status
     END IF
   END DO
   CALL LPB_terminate( data, control, info )

!  case when there are no bounded variables

   p%X_l = (/ - infty, - infty /)
   p%X_u = (/ infty, infty /)
   p%new_problem_structure = .TRUE.
   IF ( ALLOCATED( p%A%type ) ) DEALLOCATE( p%A%type )
   CALL SMT_put( p%A%type, 'SPARSE_BY_ROWS', smt_stat )
   p%A%col = (/ 1, 2 /)
   p%A%ptr = (/ 1, 3 /)
   CALL LPB_initialize( data, control )
   control%infinity = infty
   control%restore_problem = 2
!  control%print_level = 4
   DO i = 16, 16

     p%A%val = (/ 1.0_wp, 1.0_wp /)
     p%X = 0.0_wp ; p%Y = 0.0_wp ; p%Z = 0.0_wp
     CALL LPB_solve( p, data, control, info )
!    write(6,"('x=', 2ES12.4)") p%X
     IF ( info%status == 0 ) THEN
       WRITE( 6, "( I2, ':', I6, ' iterations. Optimal objective value = ',    &
     &       F6.1, ' status = ', I6 )" ) i, info%iter, info%obj, info%status
     ELSE
       WRITE( 6, "( I2, ': LPB_solve exit status = ', I6 ) " ) i, info%status
     END IF
   END DO
   CALL LPB_terminate( data, control, info )

!  case when there are no free variables

   p%X_l = (/ 0.5_wp, 0.5_wp /)
   p%X_u = (/ 0.5_wp, 0.5_wp /)
   p%new_problem_structure = .TRUE.
   IF ( ALLOCATED( p%A%type ) ) DEALLOCATE( p%A%type )
   CALL SMT_put( p%A%type, 'SPARSE_BY_ROWS', smt_stat )
   p%A%col = (/ 1, 2 /)
   p%A%ptr = (/ 1, 3 /)
   CALL LPB_initialize( data, control )
   control%infinity = infty
   control%restore_problem = 2
   DO i = 17, 17

     p%A%val = (/ 1.0_wp, 1.0_wp /)
     p%X = 0.0_wp ; p%Y = 0.0_wp ; p%Z = 0.0_wp
!    control%print_level = 1
     CALL LPB_solve( p, data, control, info )
!    write(6,"('x=', 2ES12.4)") p%X
     IF ( info%status == 0 ) THEN
       WRITE( 6, "( I2, ':', I6, ' iterations. Optimal objective value = ',    &
     &       F6.1, ' status = ', I6 )" ) i, info%iter, info%obj, info%status
     ELSE
       WRITE( 6, "( I2, ': LPB_solve exit status = ', I6 ) " ) i, info%status
     END IF
   END DO
   CALL LPB_terminate( data, control, info )
   DEALLOCATE( p%A%val, p%A%row, p%A%col )
   DEALLOCATE( p%G, p%X_l, p%X_u, p%C_l, p%C_u )
   DEALLOCATE( p%X, p%Y, p%Z, p%C )
   DEALLOCATE( p%A%type )

!  ============================
!  full test of generic problem
!  ============================

   WRITE( 6, "( /, ' full test of generic problems ', / )" )

   n = 14 ; m = 17 ; h_ne = 14 ; a_ne = 46
   ALLOCATE( p%G( n ), p%X_l( n ), p%X_u( n ) )
   ALLOCATE( p%C( m ), p%C_l( m ), p%C_u( m ) )
   ALLOCATE( p%X( n ), p%Y( m ), p%Z( n ) )
   ALLOCATE( p%A%ptr( m + 1 ) )
   ALLOCATE( p%A%val( a_ne ), p%A%row( a_ne ), p%A%col( a_ne ) )
   p%new_problem_structure = .TRUE.
   IF ( ALLOCATED( p%A%type ) ) DEALLOCATE( p%A%type )
   CALL SMT_put( p%A%type, 'COORDINATE', smt_stat )
   p%n = n ; p%m = m ; p%A%ne = a_ne 
   p%f = 1.0_wp
   p%G = (/ 0.0_wp, 2.0_wp, 0.0_wp, 0.0_wp, 2.0_wp, 0.0_wp, 2.0_wp,            &
            0.0_wp, 2.0_wp, 0.0_wp, 0.0_wp, 2.0_wp, 0.0_wp, 2.0_wp /) 
   p%C_l = (/ 4.0_wp, 2.0_wp, 6.0_wp, - infty, - infty,                        &
              4.0_wp, 2.0_wp, 6.0_wp, - infty, - infty,                        &
              - 10.0_wp, - 10.0_wp, - 10.0_wp, - 10.0_wp,                      &
              - 10.0_wp, - 10.0_wp, - 10.0_wp /)
   p%C_u = (/ 4.0_wp, infty, 10.0_wp, 2.0_wp, infty,                           &
              4.0_wp, infty, 10.0_wp, 2.0_wp, infty,                           &
              10.0_wp, 10.0_wp, 10.0_wp, 10.0_wp,                              &
              10.0_wp, 10.0_wp, 10.0_wp /)
   p%X_l = (/ 1.0_wp, 0.0_wp, 1.0_wp, 2.0_wp, - infty, - infty, - infty,       &
              1.0_wp, 0.0_wp, 1.0_wp, 2.0_wp, - infty, - infty, - infty /)
   p%X_u = (/ 1.0_wp, infty, infty, 3.0_wp, 4.0_wp, 0.0_wp, infty,             &
              1.0_wp, infty, infty, 3.0_wp, 4.0_wp, 0.0_wp, infty /)
   p%A%val = (/ 1.0_wp, 1.0_wp, 1.0_wp, 1.0_wp, 1.0_wp,                        &
                1.0_wp, 1.0_wp, 1.0_wp, 1.0_wp, 1.0_wp, 1.0_wp,                &
                1.0_wp, 1.0_wp, 1.0_wp, 1.0_wp, 1.0_wp,                        &
                1.0_wp, 1.0_wp, 1.0_wp, 1.0_wp, 1.0_wp,                        &
                1.0_wp, 1.0_wp, 1.0_wp, 1.0_wp, 1.0_wp, 1.0_wp,                &
                1.0_wp, 1.0_wp, 1.0_wp, 1.0_wp, 1.0_wp,                        &
                1.0_wp, - 1.0_wp, 1.0_wp, - 1.0_wp, 1.0_wp, - 1.0_wp,          &
                1.0_wp, - 1.0_wp, 1.0_wp, - 1.0_wp, 1.0_wp, - 1.0_wp,          &
                1.0_wp, - 1.0_wp /)
   p%A%row = (/ 1, 1, 1, 2, 2, 3, 3, 3, 3, 3, 3, 4, 4, 5, 5, 5,                &
                6, 6, 6, 7, 7, 8, 8, 8, 8, 8, 8, 9, 9, 10, 10, 10,             &
                11, 11, 12, 12, 13, 13, 14, 14, 15, 15, 16, 16, 17, 17 /)
   p%A%col = (/ 1, 3, 5, 1, 2, 1, 2, 3, 4, 5, 6, 5, 6, 2, 4, 6,                &
                8, 10, 12, 8, 9, 8, 9, 10, 11, 12, 13, 12, 13, 9, 11, 13,      &
                1, 8, 2, 9, 3, 10, 4, 11, 5, 12, 6, 13, 7, 14 /) 

   CALL LPB_initialize( data, control )
   control%infinity = infty
   control%restore_problem = 1
   control%print_level = 101
   control%itref_max = 3
   control%out = scratch_out
   control%error = scratch_out
!  control%out = 6
!  control%error = 6
   p%X = 0.0_wp ; p%Y = 0.0_wp ; p%Z = 0.0_wp
   OPEN( UNIT = scratch_out, STATUS = 'SCRATCH' )
   CALL LPB_solve( p, data, control, info )
   CLOSE( UNIT = scratch_out )
   IF ( info%status == 0 ) THEN
       WRITE( 6, "( I2, ':', I6, ' iterations. Optimal objective value = ',    &
     &       F6.1, ' status = ', I6 )" ) 1, info%iter, info%obj, info%status
   ELSE
     WRITE( 6, "( I2, ': LPB_solve exit status = ', I6 ) " ) 1, info%status
   END IF
   CALL LPB_terminate( data, control, info )
   DEALLOCATE( p%A%val, p%A%row, p%A%col )
   DEALLOCATE( p%A%ptr )
   DEALLOCATE( p%A%type )
   DEALLOCATE( p%G, p%X_l, p%X_u, p%C_l, p%C_u )
   DEALLOCATE( p%X, p%Y, p%Z, p%C )

!  Second problem

   n = 14 ; m = 17 ; h_ne = 14 ; a_ne = 46
   ALLOCATE( p%G( n ), p%X_l( n ), p%X_u( n ) )
   ALLOCATE( p%C( m ), p%C_l( m ), p%C_u( m ) )
   ALLOCATE( p%X( n ), p%Y( m ), p%Z( n ) )
   ALLOCATE( p%A%ptr( m + 1 ) )
   ALLOCATE( p%A%val( a_ne ), p%A%row( a_ne ), p%A%col( a_ne ) )
   p%new_problem_structure = .TRUE.
   IF ( ALLOCATED( p%A%type ) ) DEALLOCATE( p%A%type )
   CALL SMT_put( p%A%type, 'COORDINATE', smt_stat )
   p%n = n ; p%m = m ; p%A%ne = a_ne 
   p%f = 1.0_wp
   p%G = (/ 0.0_wp, 2.0_wp, 0.0_wp, 0.0_wp, 2.0_wp, 0.0_wp, 2.0_wp,            &
            0.0_wp, 2.0_wp, 0.0_wp, 0.0_wp, 2.0_wp, 0.0_wp, 2.0_wp /) 
   p%C_l = (/ 4.0_wp, 2.0_wp, 6.0_wp, - infty, - infty,                        &
              4.0_wp, 2.0_wp, 6.0_wp, - infty, - infty,                        &
              - 10.0_wp, - 10.0_wp, - 10.0_wp, - 10.0_wp,                      &
              - 10.0_wp, - 10.0_wp, - 10.0_wp /)
   p%C_u = (/ 4.0_wp, infty, 10.0_wp, 2.0_wp, infty,                           &
              4.0_wp, infty, 10.0_wp, 2.0_wp, infty,                           &
              10.0_wp, 10.0_wp, 10.0_wp, 10.0_wp,                              &
              10.0_wp, 10.0_wp, 10.0_wp /)
   p%X_l = (/ 1.0_wp, 0.0_wp, 1.0_wp, 2.0_wp, - infty, - infty, - infty,       &
              1.0_wp, 0.0_wp, 1.0_wp, 2.0_wp, - infty, - infty, - infty /)
   p%X_u = (/ 1.0_wp, infty, infty, 3.0_wp, 4.0_wp, 0.0_wp, infty,             &
              1.0_wp, infty, infty, 3.0_wp, 4.0_wp, 0.0_wp, infty /)
   p%A%val = (/ 1.0_wp, 1.0_wp, 1.0_wp, 1.0_wp, 1.0_wp,                        &
                1.0_wp, 1.0_wp, 1.0_wp, 1.0_wp, 1.0_wp, 1.0_wp,                &
                1.0_wp, 1.0_wp, 1.0_wp, 1.0_wp, 1.0_wp,                        &
                1.0_wp, 1.0_wp, 1.0_wp, 1.0_wp, 1.0_wp,                        &
                1.0_wp, 1.0_wp, 1.0_wp, 1.0_wp, 1.0_wp, 1.0_wp,                &
                1.0_wp, 1.0_wp, 1.0_wp, 1.0_wp, 1.0_wp,                        &
                1.0_wp, - 1.0_wp, 1.0_wp, - 1.0_wp, 1.0_wp, - 1.0_wp,          &
                1.0_wp, - 1.0_wp, 1.0_wp, - 1.0_wp, 1.0_wp, - 1.0_wp,          &
                1.0_wp, - 1.0_wp /)
   p%A%row = (/ 1, 1, 1, 2, 2, 3, 3, 3, 3, 3, 3, 4, 4, 5, 5, 5,                &
                6, 6, 6, 7, 7, 8, 8, 8, 8, 8, 8, 9, 9, 10, 10, 10,             &
                11, 11, 12, 12, 13, 13, 14, 14, 15, 15, 16, 16, 17, 17 /)
   p%A%col = (/ 1, 3, 5, 1, 2, 1, 2, 3, 4, 5, 6, 5, 6, 2, 4, 6,                &
                8, 10, 12, 8, 9, 8, 9, 10, 11, 12, 13, 12, 13, 9, 11, 13,      &
                1, 8, 2, 9, 3, 10, 4, 11, 5, 12, 6, 13, 7, 14 /) 

   CALL LPB_initialize( data, control )
   control%infinity = infty
   control%restore_problem = 0
   control%treat_zero_bounds_as_general = .TRUE.
   p%X = 0.0_wp ; p%Y = 0.0_wp ; p%Z = 0.0_wp
   CALL LPB_solve( p, data, control, info )
   IF ( info%status == 0 ) THEN
       WRITE( 6, "( I2, ':', I6, ' iterations. Optimal objective value = ',    &
     &       F6.1, ' status = ', I6 )" ) 2, info%iter, info%obj, info%status
   ELSE
     WRITE( 6, "( I2, ': LPB_solve exit status = ', I6 ) " ) 2, info%status
   END IF
   CALL LPB_terminate( data, control, info )
   DEALLOCATE( p%A%val, p%A%row, p%A%col )
   DEALLOCATE( p%A%ptr )
   DEALLOCATE( p%G, p%X_l, p%X_u, p%C_l, p%C_u )
   DEALLOCATE( p%X, p%Y, p%Z, p%C )
   DEALLOCATE( p%A%type )
   IF ( ALLOCATED( p%WEIGHT ) ) DEALLOCATE( p%WEIGHT )
   IF ( ALLOCATED( p%X0 ) ) DEALLOCATE( p%X0 )

   END PROGRAM GALAHAD_LPB_EXAMPLE
