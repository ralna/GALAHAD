! THIS VERSION: GALAHAD 2.6 - 09/06/2014 AT 08:30 GMT.
   PROGRAM GALAHAD_CONVERT_EXAMPLE
   USE GALAHAD_CONVERT_double         ! double precision version
   IMPLICIT NONE
   INTEGER, PARAMETER :: wp = KIND( 1.0D+0 ) ! set precision
   TYPE ( SMT_type ) :: A, A_by_cols
   TYPE ( CONVERT_control_type ) :: control        
   TYPE ( CONVERT_inform_type ) :: inform
   INTEGER :: i, j, mode, s, type
   INTEGER, PARAMETER :: len_iw = 0, len_w = 0
   INTEGER, DIMENSION( len_iw ) :: IW
   REAL ( KIND = wp ), DIMENSION( len_w ) :: W
! set problem data
   DO mode = 1, 2
     control%order = .TRUE.
     IF ( mode == 2 ) THEN
       WRITE( 6, "( /, ' construct the transpose' )" )
       control%transpose = .TRUE.
     END IF
     DO type = 1, 5
       A%m = 4 ; A%n = 5 ; A%ne = 9
       SELECT CASE ( type )
       CASE ( 1 ) ! dense
         CALL SMT_put( A%type, 'DENSE', s )
         ALLOCATE( A%val( A%m * A%n ) ) 
         A%val = (/ 11.0_wp, 0.0_wp, 13.0_wp, 0.0_wp, 15.0_wp,                 &
                    0.0_wp, 22.0_wp, 0.0_wp, 24.0_wp, 0.0_wp,                  &
                    0.0_wp, 32.0_wp, 33.0_wp, 0.0_wp, 0.0_wp,                  &
                    0.0_wp, 0.0_wp, 0.0_wp, 44.0_wp, 45.0_wp /)
       CASE ( 2 ) ! dense by columns
         CALL SMT_put( A%type, 'DENSE_BY_COLUMNS', s )
         ALLOCATE( A%val( A%m * A%n ) ) 
         A%val = (/ 11.0_wp, 0.0_wp, 0.0_wp, 0.0_wp,                           &
                    0.0_wp, 22.0_wp, 32.0_wp, 0.0_wp,                          &
                    13.0_wp, 0.0_wp, 33.0_wp, 0.0_wp,                          &
                    0.0_wp, 24.0_wp, 0.0_wp, 44.0_wp,                          &
                    15.0_wp, 0.0_wp, 0.0_wp, 45.0_wp /)
       CASE ( 3 ) ! sparse by rows
         CALL SMT_put( A%type, 'SPARSE_BY_ROWS', s )
         ALLOCATE( A%ptr( A%m + 1 ), A%col( A%ne ), A%val( A%ne ) ) 
         A%ptr = (/ 1, 4, 6, 8, 10 /)
         A%col = (/ 1, 5, 3, 2, 4, 3, 2, 4, 5 /)
         A%val = (/ 11.0_wp, 15.0_wp, 13.0_wp, 22.0_wp, 24.0_wp,               &
                    33.0_wp, 32.0_wp, 44.0_wp, 45.0_wp /)
       CASE ( 4 ) ! sparse by columns
         CALL SMT_put( A%type, 'SPARSE_BY_COLUMNS', s )
         ALLOCATE( A%ptr( A%n + 1 ), A%row( A%ne ), A%val( A%ne ) ) 
         A%ptr = (/ 1, 2, 4, 6, 8, 10 /)
         A%row = (/ 1, 3, 2, 1, 3, 2, 4, 4, 1 /)
         A%val = (/ 11.0_wp, 32.0_wp, 22.0_wp, 13.0_wp, 33.0_wp,               &
                    24.0_wp, 44.0_wp, 45.0_wp, 15.0_wp /)
       CASE ( 5 ) ! sparse co-ordinate
         CALL SMT_put( A%type, 'COORDINATE', s )
         ALLOCATE( A%row( A%ne ), A%col( A%ne ), A%val( A%ne ) ) 
         A%row = (/ 4, 1, 3, 2, 1, 3, 4, 2, 1 /)
         A%col = (/ 5, 1, 2, 2, 3, 3, 4, 4, 5 /)
         A%val = (/ 45.0_wp, 11.0_wp, 32.0_wp, 22.0_wp, 13.0_wp, 33.0_wp,      &
                    44.0_wp, 24.0_wp, 15.0_wp /)
       END SELECT
       CALL CONVERT_to_column_format( A, A_by_cols, control, inform,           &
                                      IW, len_iw, W, len_w ) ! convert
       WRITE( 6, "( /, ' convert from ', A, ' to column format ')" )           &
         SMT_get( A%type )
       DO i = 1, A_by_cols%n
         WRITE( 6, "( ' column ', I0, ', (row,value)' )" ) i
         WRITE( 6, "( 3( I2, F5.1 ) )" )                                       &
           ( A_by_cols%row( j ), A_by_cols%val( j ),                           &
             j = A_by_cols%ptr( i ), A_by_cols%ptr( i + 1 ) - 1 )
       END DO
       SELECT CASE ( type ) ! deallocate space
       CASE ( 1, 2 ) ! dense + dense by columns
         DEALLOCATE( A%val )
       CASE ( 3 ) ! sparse by rows
         DEALLOCATE( A%ptr, A%col, A%val )
       CASE ( 4 ) ! sparse by columns
         DEALLOCATE( A%ptr, A%row, A%val )
       CASE ( 5 ) ! sparse co-ordinate
         DEALLOCATE( A%row, A%col, A%val ) 
       END SELECT
       DEALLOCATE( A_by_cols%ptr, A_by_cols%row, A_by_cols%val )
     END DO
   END DO
   END PROGRAM GALAHAD_CONVERT_EXAMPLE

