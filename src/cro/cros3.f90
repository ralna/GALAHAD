! THIS VERSION: GALAHAD 4.1 - 11/05/2022 AT 14:30 GMT.
   PROGRAM GALAHAD_CRO_EXAMPLE3 ! AFIRO example from Netlib LP set
   USE GALAHAD_CRO_double       ! double precision version
   IMPLICIT NONE
   INTEGER, PARAMETER :: wp = KIND( 1.0D+0 ) ! set precision
   REAL ( KIND = wp ), PARAMETER :: infinity = 10.0_wp ** 20
   TYPE ( CRO_data_type ) :: data
   TYPE ( CRO_control_type ) :: control
   TYPE ( CRO_inform_type ) :: inform
   INTEGER :: i
   INTEGER, PARAMETER :: n = 32, m = 27, m_equal = 8, a_ne = 83, h_ne = 0
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
   H_ptr = 0 ! pointers to H columns (this is AFIRO, a linear program)
   g = 0.0_wp ! objective gradient
   G( 2 ) = - 0.4_wp ; g( 13 ) = - 0.32_wp ; G( 17 ) = - 0.6_wp
   G( 29 ) = - 0.48_wp ; g( 32 ) = 10.0_wp
   A_val  = (/  1.0_wp, -1.0_wp,  1.0_wp, -1.06_wp,  1.0_wp,                   &
                1.0_wp, -1.0_wp,  1.4_wp,  1.0_wp, -1.0_wp,                    &
                1.0_wp, -1.0_wp, -1.0_wp, -1.0_wp,  1.0_wp,                    &
               -0.86_wp, -0.96_wp, -1.06_wp, -1.06_wp,  1.0_wp,                &
               -1.0_wp, -1.0_wp,  1.0_wp,  1.0_wp, -1.0_wp,                    &
               -1.0_wp,  1.0_wp,  1.0_wp, -1.0_wp,  1.0_wp,                    &
                1.0_wp, -0.43_wp,  1.0_wp,  1.0_wp,  1.40_wp,                  &
               -1.0_wp, -0.43_wp,  1.0_wp, -0.37_wp, -0.43_wp,                 &
               -0.39_wp,  1.0_wp,  1.0_wp,  1.0_wp,  1.0_wp,                   &
                1.0_wp,  1.0_wp, -1.0_wp, -1.0_wp,  1.0_wp,                    &
               -1.0_wp,  1.0_wp,  1.0_wp, -1.0_wp,  1.0_wp,                    &
               -1.0_wp,  2.364_wp,  2.249_wp,  2.408_wp,  2.219_wp,            &
                2.429_wp,  2.386_wp,  2.279_wp, -1.0_wp,  2.191_wp,            &
                0.109_wp, -1.0_wp,  0.1_wp,  0.10_wp, -1.0_wp,                 &
                0.108_wp, 0.109_wp,  0.301_wp, -1.0_wp,  0.313_wp,             &
                0.313_wp,  0.301_wp,  0.326_wp, -1.0_wp,  1.0_wp,              &
                1.0_wp,  1.0_wp,  1.0_wp /) ! A values
   A_col = (/ 3, 1, 2, 1, 4, 1, 2, 13, 13, 7,                                  &
              14, 6, 5, 8, 15, 8, 7, 5, 6, 5,                                  &
              9, 10, 6, 7, 11, 12, 8, 18, 16, 19,                              &
              17, 16, 20, 16, 29, 17, 21, 31, 24, 22,                          &
              23, 30, 32, 23, 21, 24, 22, 29, 25, 21,                          &
              26, 22, 23, 27, 24, 28, 9, 27, 11, 26,                           &
              12, 10, 28, 19, 25, 16, 3, 22, 24, 14,                           &
              23, 21, 1, 18, 7, 6, 5, 8, 30, 4, 20, 31, 15 /) ! A columns
   A_ptr = (/ 1, 4, 6, 7, 9, 15, 20, 22, 24, 26, 28, 32, 34, 35, 37, 42, 49,   &
              51, 53, 55, 57, 66, 68, 73, 75, 80, 82, 84/) ! pointers to A cols
   X_l = 0.0_wp ! variable lower bound
   X_u = infinity ! variable upper bound
   C_l = - infinity ! constraint lower bound
   C_l( 1 ) = 0.0_wp ; C_l( 2 ) = 0.0_wp
   C_l( 5 ) = 0.0_wp ; C_l( 6 ) = 0.0_wp
   C_l( 11 ) = 0.0_wp ; C_l( 12 ) = 0.0_wp
   C_l( 15 ) = 0.0_wp ; C_l( 16 ) = 44.0_wp
   C_u = 0.0_wp ! constraint upper bound
   C_u( 3 ) = 80.0_wp ; C_u( 7 ) = 80.0_wp
   C_u( 13 ) = 500.0_wp ; C_u( 16 ) = 44.0_wp
   C_u( 17 ) = 500.0_wp ; C_u( 26 ) = 310.0_wp
   C_u( 27 ) = 300.0_wp
   C = (/  0.0_wp, 0.0_wp, 80.0_wp, 0.0_wp, 0.0_wp, 0.0_wp, 18.214_wp, 0.0_wp, &
           0.0_wp, 0.0_wp, 0.0_wp, 0.0_wp, 500.0_wp, 0.0_wp, 0.0_wp, 440.0_wp, &
           0.0_wp, 0.0_wp, 0.0_wp, 0.0_wp, 0.0_wp, 0.0_wp, 0.0_wp, 0.0_wp,     &
           -378.46_wp, 299.80_wp, 19.307_wp /) ! optimal constraint value
   X = (/  80.0_wp, 25.5_wp, 54.5_wp, 84.8_wp, 18.214_wp, 0.0_wp, 0.0_wp,      &
           0.0_wp, 0.0_wp, 0.0_wp, 0.0_wp, 0.0_wp, 18.214_wp, 0.0_wp,          &
           19.307_wp, 500.0_wp, 475.92_wp, 24.08_wp, 0.0_wp, 215.0_wp, 0.0_wp, &
           0.0_wp, 0.0_wp, 0.0_wp, 0.0_wp, 0.0_wp, 0.0_wp, 0.0_wp, 339.94_wp,  &
           383.94_wp, 0.0_wp, 0.0_wp /) ! optimal variables
   Y = (/ -0.62857_wp, 0.0_wp, -0.34477_wp, -0.22857_wp, 0.0_wp, 0.0_wp,       &
          0.0_wp, 0.0_wp, 0.0_wp, 0.0_wp, -0.94286_wp, 0.0_wp, -0.87434_wp,    &
          -0.34286_wp, 0.0_wp, 0.0_wp, 0.0_wp, 0.0_wp, 0.0_wp, 0.0_wp,         &
          0.0_wp, -0.62857_wp, 0.0_wp, -0.94286_wp, 0.0_wp, 0.0_wp,            &
          0.0_wp /) ! optimal Lagrange multipliers
   Z = (/  0.0_wp, 0.0_wp, 0.0_wp, 0.0_wp, 0.0_wp, 0.0_wp, 0.0_wp, 0.0_wp,     &
           0.0_wp, 0.0_wp, 0.0_wp, 0.0_wp, 0.0_wp, 0.0_wp, 0.0_wp, 0.0_wp,     &
           0.0_wp, 0.0_wp, 0.94286_wp, 0.0_wp, 0.0_wp, 0.0_wp, 0.0_wp, 0.0_wp, &
           0.0_wp, 0.0_wp, 0.0_wp, 0.0_wp, 0.0_wp, 0.0_wp, 0.0_wp,             &
           10.0_wp /) ! optimal dual variables
   C_stat = (/ -1, -1, 1, 1, -1, -1, 0, 1, 1, 1, -1, -1, 1, 1, -1, -1, 0, 1,   &
               1, 1, 1, 1, 1, 1, 0, 0, 0 /)    ! constraint status
   X_stat = (/ 0, 0, 0, 0, 0, -1, -1, -1, -1, -1, -1, -1, 0, -1, 0, 0, 0, 0,   &
               -1, 0, -1, -1, -1, -1, -1, -1, -1, -1, 0, 0,                    &
               -1, -1 /) ! variable status
! problem data complete
   CALL CRO_initialize( data, control, inform ) ! Initialize control parameters
   control%print_level = 101
   control%symmetric_linear_solver = 'ma57'
   control%unsymmetric_linear_solver = 'ma48'

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
   END PROGRAM GALAHAD_CRO_EXAMPLE3














