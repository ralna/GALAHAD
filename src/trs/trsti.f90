! THIS VERSION: GALAHAD 3.3 - 13/12/2021 AT 11:45 GMT.
   PROGRAM GALAHAD_TRS_interface_test
   USE GALAHAD_TRS_DOUBLE                            ! double precision version
   IMPLICIT NONE
   INTEGER, PARAMETER :: wp = KIND( 1.0D+0 ) ! set precision
   REAL ( KIND = wp ), PARAMETER :: one = 1.0_wp, two = 2.0_wp
   INTEGER :: status, storage_type, a_is, m_is
   LOGICAL :: use_a, use_m
   TYPE ( TRS_full_data_type ) :: data
   TYPE ( TRS_control_type ) :: control
   TYPE ( TRS_inform_type ) :: inform
   CHARACTER ( len = 1 ) :: st
   CHARACTER ( len = 2 ) :: ma
   INTEGER, PARAMETER :: n = 3, m = 1, h_ne = 4, m_ne = 3, a_ne = 3
   INTEGER, PARAMETER :: h_dense_ne = n * ( n + 1 ) / 2, m_dense_ne = h_dense_ne
   REAL ( KIND = wp ) :: f = 0.96_wp, radius = 1.0_wp
   REAL ( KIND = wp ), DIMENSION( n ) :: X
   INTEGER, DIMENSION( 0 ) :: null
   REAL ( KIND = wp ), DIMENSION( n ) :: C = (/ 0.0_wp, 2.0_wp, 0.0_wp /)
   INTEGER, DIMENSION( h_ne ) :: H_row = (/ 1, 2, 3, 3 /)
   INTEGER, DIMENSION( h_ne ) :: H_col = (/ 1, 2, 3, 1 /)
   INTEGER, DIMENSION( n + 1 ) :: H_ptr = (/ 1, 2, 3, 5 /)
   REAL ( KIND = wp ), DIMENSION( h_ne ) ::                                    &
     H_val = (/ 1.0_wp, 2.0_wp, 3.0_wp, 4.0_wp /)
   INTEGER, DIMENSION( m_ne ) :: M_row = (/ 1, 2, 3 /)
   INTEGER, DIMENSION( m_ne ) :: M_col = (/ 1, 2, 3 /)
   INTEGER, DIMENSION( n + 1 ) :: M_ptr = (/ 1, 2, 3, 4 /)
   REAL ( KIND = wp ), DIMENSION( m_ne ) ::                                    &
     M_val = (/ 1.0_wp, 2.0_wp, 1.0_wp /)
   INTEGER, DIMENSION( a_ne ) :: A_row = (/ 1, 1, 1 /)
   INTEGER, DIMENSION( a_ne ) :: A_col = (/ 1, 2, 3 /)
   INTEGER, DIMENSION( m + 1 ) :: A_ptr = (/ 1, 4 /)
   REAL ( KIND = wp ), DIMENSION( a_ne ) ::                                    &
     A_val = (/ 1.0_wp, 1.0_wp, 1.0_wp /)
   REAL ( KIND = wp ), DIMENSION( h_dense_ne ) ::                              &
     H_dense_val = (/ 1.0_wp, 0.0_wp, 2.0_wp, 4.0_wp, 0.0_wp, 3.0_wp /)
   REAL ( KIND = wp ), DIMENSION( m_dense_ne ) ::                              &
     M_dense_val = (/ 1.0_wp, 0.0_wp, 2.0_wp, 0.0_wp, 0.0_wp, 1.0_wp /)
   REAL ( KIND = wp ), DIMENSION( n ) ::                                       &
     H_diag_val = (/ 1.0_wp, 0.0_wp, 2.0_wp /)
   REAL ( KIND = wp ), DIMENSION( n ) ::                                       &
     M_diag_val = (/ 1.0_wp, 2.0_wp, 1.0_wp /)

!  =====================================
!  basic test of various storage formats
!  =====================================

   WRITE( 6, "( /, ' ==== basic tests of storage formats ===== ', / )" )

    DO a_is = 0, 1
     use_a = a_is == 1  ! add a linear constraint?
     DO m_is = 0, 1
       use_m = m_is == 1 ! include a scaling matrix?
       IF ( use_a .AND. use_m ) THEN
         ma = 'MA'
       ELSE IF ( use_a ) THEN
         ma = 'A '
       ELSE IF ( use_m ) THEN
         ma = 'M '
       ELSE
         ma = '  '
       END IF
       DO storage_type = 1, 4 ! loop over a variety of storage types
         CALL TRS_initialize( data, control, inform )
         SELECT CASE ( storage_type )
         CASE ( 1 ) ! sparse co-ordinate storage
           st = 'C'
           CALL TRS_import( control, data, status, n, 'COORDINATE',            &
                            H_ne, H_row, H_col, null )
           IF ( use_m ) CALL TRS_import_M( data, status, 'COORDINATE',         &
                                           M_ne, M_row, M_col, null )
           IF ( use_a ) CALL TRS_import_A( data, status, m, 'COORDINATE',      &
                                           A_ne, A_row, A_col, null )
           IF ( use_a .AND. use_m ) THEN
             CALL TRS_solve_problem( data, status, radius, f, C, H_val, X,     &
                                     M_val = M_val, A_val = A_val )
           ELSE IF ( use_a ) THEN
             CALL TRS_solve_problem( data, status, radius, f, C, H_val, X,     &
                                     A_val = A_val )
           ELSE IF ( use_m ) THEN
             CALL TRS_solve_problem( data, status, radius, f, C, H_val, X,     &
                                     M_val = M_val )
           ELSE
             CALL TRS_solve_problem( data, status, radius, f, C, H_val, X )
           END IF
         CASE ( 2 ) ! sparse row-wise storage
           st = 'R'
           CALL TRS_import( control, data, status, n, 'SPARSE_BY_ROWS',        &
                            H_ne, null, H_col, H_ptr )
           IF ( use_m ) CALL TRS_import_M( data, status, 'SPARSE_BY_ROWS',     &
                                           M_ne, null, M_col, M_ptr )
           IF ( use_a ) CALL TRS_import_A( data, status, m, 'SPARSE_BY_ROWS',  &
                                           A_ne, null, A_col, A_ptr )
           IF ( use_a .AND. use_m ) THEN
             CALL TRS_solve_problem( data, status, radius, f, C, H_val, X,     &
                                     M_val = M_val, A_val = A_val )
           ELSE IF ( use_a ) THEN
             CALL TRS_solve_problem( data, status, radius, f, C, H_val, X,     &
                                     A_val = A_val )
           ELSE IF ( use_m ) THEN
             CALL TRS_solve_problem( data, status, radius, f, C, H_val, X,     &
                                     M_val = M_val )
           ELSE
             CALL TRS_solve_problem( data, status, radius, f, C, H_val, X )
           END IF
         CASE ( 3 ) ! dense storage
           st = 'D'
           CALL TRS_import( control, data, status, n, 'DENSE',                 &
                            H_ne, null, null, null )
           IF ( use_m ) CALL TRS_import_M( data, status, 'DENSE',              &
                                           M_ne, null, null, null )
           IF ( use_a ) CALL TRS_import_A( data, status, m, 'DENSE',           &
                                           A_ne, null, null, null )
           IF ( use_a .AND. use_m ) THEN
             CALL TRS_solve_problem( data, status, radius, f, C,               &
                                     H_dense_val, X,                           &
                                     M_val = M_dense_val, A_val = A_val )
           ELSE IF ( use_a ) THEN
             CALL TRS_solve_problem( data, status, radius, f, C,               &
                                     H_dense_val, X, A_val = A_val )
           ELSE IF ( use_m ) THEN
             CALL TRS_solve_problem( data, status, radius, f, C,               &
                                     H_dense_val, X, M_val = M_dense_val )
           ELSE
             CALL TRS_solve_problem( data, status, radius, f, C,               &
                                     H_dense_val, X )
           END IF
         CASE ( 4 ) ! diagonal H, M, dense A
           st = 'I'
           CALL TRS_import( control, data, status, n, 'DIAGONAL',              &
                            H_ne, null, null, null )
           IF ( use_m ) CALL TRS_import_M( data, status, 'DIAGONAL',           &
                                           M_ne, null, null, null )
           IF ( use_a ) CALL TRS_import_A( data, status, m, 'DIAGONAL',        &
                                           A_ne, null, null, null )
           IF ( use_a .AND. use_m ) THEN
             CALL TRS_solve_problem( data, status, radius, f, C,               &
                                     H_diag_val, X,                            &
                                     M_val = M_diag_val, A_val = A_val )
           ELSE IF ( use_a ) THEN
             CALL TRS_solve_problem( data, status, radius, f, C,               &
                                     H_diag_val, X, A_val = A_val )
           ELSE IF ( use_m ) THEN
             CALL TRS_solve_problem( data, status, radius, f, C,               &
                                     H_diag_val, X, M_val = M_diag_val )
           ELSE
             CALL TRS_solve_problem( data, status, radius, f, C,               &
                                     H_diag_val, X )
           END IF
         END SELECT
         CALL TRS_information( data, inform, status )
         WRITE( 6, "( ' format ', A1, A2, ':',                                 &
        &  ' TRS_solve_problem exit status = ', I4, ', f = ', f0.2 )" )        &
          st, ma, status, inform%obj
       END DO
     END DO
     CALL TRS_terminate( data, control, inform ) !  delete internal workspace
   END DO

   STOP
   END PROGRAM GALAHAD_TRS_interface_test
