! THIS VERSION: GALAHAD 3.3 - 28/05/2021 AT 13:30 GMT.
   PROGRAM GALAHAD_CRO_TEST   ! ** to be improved!
   USE GALAHAD_CRO_double                    ! double precision version
   IMPLICIT NONE
   INTEGER, PARAMETER :: wp = KIND( 1.0D+0 ) ! set precision
   REAL ( KIND = wp ), PARAMETER :: infinity = 10.0_wp ** 20
   TYPE ( CRO_data_type ) :: data
   TYPE ( CRO_control_type ) :: control        
   TYPE ( CRO_inform_type ) :: inform
   INTEGER :: i
   INTEGER, PARAMETER :: n = 11, m = 3, m_equal = 1, a_ne = 30, h_ne = 21
   INTEGER, DIMENSION( h_ne ) :: H_col
   INTEGER, DIMENSION( n + 1 ) :: H_ptr
   REAL ( KIND = wp ), DIMENSION( h_ne ) :: H_val
   INTEGER, DIMENSION( a_ne ) :: A_col
   INTEGER, DIMENSION( m + 1 ) :: A_ptr
   REAL ( KIND = wp ), DIMENSION( a_ne ) :: A_val
   REAL ( KIND = wp ), DIMENSION( n ) :: G, X_l, X_u, X, Z
   REAL ( KIND = wp ), DIMENSION( m ) :: C_l, C_u, C, Y
   INTEGER, DIMENSION( m ) :: C_stat
   INTEGER, DIMENSION( n ) :: X_stat
! start problem data
   H_val = (/ 1.0D+0, 5.0D-1, 1.0D+0, 5.0D-1, 1.0D+0, 5.0D-1, 1.0D+0, 5.0D-1,  &
              1.0D+0, 5.0D-1, 1.0D+0, 5.0D-1, 1.0D+0, 5.0D-1, 1.0D+0, 5.0D-1,  &
              1.0D+0, 5.0D-1, 1.0D+0, 5.0D-1, 1.0D+0 /) ! H values
   H_col = (/ 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10,    &
              11 /)                              ! H columns
   H_ptr = (/ 1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22 /) ! pointers to H col
   A_val  = (/ 1.0D+0, 1.0D+0, 1.0D+0, 1.0D+0, 1.0D+0, 1.0D+0, 1.0D+0, 1.0D+0, &
               1.0D+0, 1.0D+0, 1.0D+0, 1.0D+0, 1.0D+0, 1.0D+0, 1.0D+0, 1.0D+0, &
               1.0D+0, 1.0D+0, 1.0D+0, 1.0D+0, 1.0D+0, 1.0D+0, 1.0D+0, 1.0D+0, &
               1.0D+0, 1.0D+0, 1.0D+0, 1.0D+0, 1.0D+0, 1.0D+0 /) ! A values
   A_col = (/ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 3, 4, 5, 6, 7, 8, 9, 10, 11,  &
              2, 3, 4, 5, 6, 7, 8, 9, 10, 11 /) ! A columns
   A_ptr = (/ 1, 12, 21, 31 /)                  ! pointers to A columns
   G   = (/ 5.0D-1, -5.0D-1, -1.0D+0, -1.0D+0, -1.0D+0,  -1.0D+0, -1.0D+0,     &
           -1.0D+0, -1.0D+0, -1.0D+0, -5.0D-1 /) ! objective gradient
   C_l = (/  1.0D+1, 9.0D+0, - infinity /)       ! constraint lower bound
   C_u = (/  1.0D+1, infinity, 1.0D+1 /)         ! constraint upper bound
   X_l = (/ 0.0D+0, 1.0D+0, 1.0D+0, 1.0D+0, 1.0D+0, 1.0D+0,                    &
            1.0D+0, 1.0D+0, 1.0D+0, 1.0D+0, 1.0D+0 /) ! variable lower bound
   X_u  = (/ infinity, infinity, infinity, infinity, infinity, infinity,       &
             infinity, infinity, infinity, infinity, infinity /) ! upper bound
   C = (/ 1.0D+1, 9.0D+0, 1.0D+1 /)              ! optimal constraint value
   X = (/ 0.0D+0, 1.0D+0, 1.0D+0, 1.0D+0, 1.0D+0, 1.0D+0, 1.0D+0, 1.0D+0,      &
          1.0D+0, 1.0D+0, 1.0D+0 /)              ! optimal variables
   Y = (/  -1.0D+0, 1.5D+0, -2.0D+0 /)           ! optimal Lagrange multipliers
   Z = (/ 2.0D+0, 4.0D+0, 2.5D+0, 2.5D+0, 2.5D+0, 2.5D+0,                      &
          2.5D+0, 2.5D+0, 2.5D+0, 2.5D+0, 2.5D+0 /) ! optimal dual variables
   C_stat = (/ -1, -1, 1 /)                         ! constraint status
   X_stat = (/ -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 /) ! variable status
! problem data complete
   CALL CRO_initialize( data, control, inform ) ! Initialize control parameters
   CALL CRO_crossover( n, m, m_equal, H_val, H_col, H_ptr, A_val, A_col,       &
                       A_ptr, G, C_l, C_u, X_l, X_u, C, X, Y, Z, C_stat,       &
                       X_stat , data, control, inform )  ! crossover
   IF ( inform%status == 0 ) THEN                   ! successful return
     WRITE( 6, "( '      x_l          x          x_u          z     stat', /,  &
   &               ( 4ES12.4, I5 ) )" )                                        &
       ( X_l( i ), X( i ), X_u( i ), Z( i ), X_stat( i ), i = 1, n )
     WRITE( 6, "( '      c_l          c          c_u          y     stat', /,  &
   &               ( 4ES12.4, I5 ) )" )                                        &
       ( C_l( i ), C( i ), C_u( i ), Y( i ), C_stat( i ), i = 1, m )
     WRITE( 6, "( ' CRO_solve exit status = ', I0 ) " ) inform%status
   ELSE                                            ! error returns
     WRITE( 6, "( ' CRO_solve exit status = ', I0 ) " ) inform%status
   END IF
   CALL CRO_terminate( data, control, inform )     ! delete internal workspace
   END PROGRAM GALAHAD_CRO_TEST

