! THIS VERSION: GALAHAD 3.3 - 24/12/2021 AT 12:50 GMT.
   PROGRAM GALAHAD_LPB_interface_test
   USE GALAHAD_LPB_double                       ! double precision version
   USE GALAHAD_SYMBOLS
   IMPLICIT NONE
   INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )    ! set precision
   REAL ( KIND = wp ), PARAMETER :: infinity = 10.0_wp ** 20
   TYPE ( LPB_control_type ) :: control
   TYPE ( LPB_inform_type ) :: inform
   TYPE ( LPB_full_data_type ) :: data
   INTEGER :: n, m, A_ne
   INTEGER :: data_storage_type, status
   REAL ( KIND = wp ) :: f
   REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: X, Z, X_l, X_u, G
   REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: Y, C, C_l, C_u
   INTEGER, ALLOCATABLE, DIMENSION( : ) :: A_row, A_col, A_ptr
   REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: A_val, A_dense
   INTEGER, ALLOCATABLE, DIMENSION( : ) :: C_stat, X_stat
   CHARACTER ( len = 2 ) :: st

! set up problem data

   n = 3 ;  m = 2 ; A_ne = 4
   f = 1.0_wp
   ALLOCATE( X( n ), Z( n ), X_l( n ), X_u( n ), G( n ), X_stat( n ) )
   ALLOCATE( C( m ), Y( m ), C_l( m ), C_u( m ), C_stat( m ) )
   G = (/ 0.0_wp, 2.0_wp, 0.0_wp /)         ! objective gradient
   C_l = (/ 1.0_wp, 2.0_wp /)               ! constraint lower bound
   C_u = (/ 2.0_wp, 2.0_wp /)               ! constraint upper bound
   X_l = (/ - 1.0_wp, - infinity, - infinity /) ! variable lower bound
   X_u = (/ 1.0_wp, infinity, 2.0_wp /)     ! variable upper bound
   ALLOCATE( A_val( A_ne ), A_row( A_ne ), A_col( A_ne ), A_ptr( m + 1 ) )
   ALLOCATE( A_dense( m * n ) )
   A_val = (/ 2.0_wp, 1.0_wp, 1.0_wp, 1.0_wp /)
   A_row = (/ 1, 1, 2, 2 /)
   A_col = (/ 1, 2, 2, 3 /)
   A_ptr = (/ 1, 3, 5 /)
   A_dense = (/ 2.0_wp, 1.0_wp, 0.0_wp, 0.0_wp, 1.0_wp, 1.0_wp /)

! problem data complete

!  =====================================
!  basic test of various storage formats
!  =====================================

   WRITE( 6, "( /, ' basic tests of lp storage formats', / )" )

   DO data_storage_type = 1, 3
     CALL LPB_initialize( data, control, inform )
     X = 0.0_wp ; Y = 0.0_wp ; Z = 0.0_wp ! start from zero
     SELECT CASE ( data_storage_type )
     CASE ( 1 ) ! sparse co-ordinate storage
       st = ' C'
       CALL LPB_import( control, data, status, n, m,                           &
                        'coordinate', A_ne, A_row, A_col, A_ptr )
       CALL LPB_solve_lp( data, status, G, f, A_val, C_l, C_u,                 &
                          X_l, X_u, X, C, Y, Z, X_stat, C_stat )
     CASE ( 2 ) ! sparse by rows
       st = ' R'
       CALL LPB_import( control, data, status, n, m,                           &
                        'sparse_by_rows', A_ne, A_row, A_col, A_ptr )
       CALL LPB_solve_lp( data, status, G, f, A_val, C_l, C_u,                 &
                          X_l, X_u, X, C, Y, Z, X_stat, C_stat )
     CASE ( 3 ) ! dense
       st = ' D'
       CALL LPB_import( control, data, status, n, m,                           &
                        'dense', A_ne, A_row, A_col, A_ptr )
       CALL LPB_solve_lp( data, status, G, f, A_dense, C_l, C_u,               &
                          X_l, X_u, X, C, Y, Z, X_stat, C_stat )
     END SELECT
     CALL LPB_information( data, inform, status )
     IF ( inform%status == 0 ) THEN
       WRITE( 6, "( A2, ':', I6, ' iterations. Optimal objective value = ',    &
     &    F5.2, ' status = ', I0 )" ) st, inform%iter, inform%obj, inform%status
     ELSE
       WRITE( 6, "( A2, ': LPB_solve exit status = ', I0 ) " ) st, inform%status
     END IF
     CALL LPB_terminate( data, control, inform )  ! delete internal workspace
   END DO
   DEALLOCATE( X, C, G, Y, Z, X_l, X_u, C_l, C_u, X_stat, C_stat )
   DEALLOCATE( A_val, A_row, A_col, A_ptr, A_dense )

   END PROGRAM GALAHAD_LPB_interface_test
x
