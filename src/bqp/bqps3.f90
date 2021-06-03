! THIS VERSION: GALAHAD 3.3 - 03/06/2021 AT 08:15 GMT.
   PROGRAM GALAHAD_BQP_THIRD_EXAMPLE
   USE GALAHAD_BQP_double         ! double precision version
   IMPLICIT NONE
   INTEGER, PARAMETER :: wp = KIND( 1.0D+0 ) ! set precision
   REAL ( KIND = wp ), PARAMETER :: infinity = 10.0_wp ** 20
   TYPE ( QPT_problem_type ) :: p
   TYPE ( BQP_reverse_type ) :: reverse
   TYPE ( BQP_data_type ) :: data
   TYPE ( BQP_control_type ) :: control        
   TYPE ( BQP_inform_type ) :: inform
   TYPE ( GALAHAD_userdata_type ) :: userdata
   INTEGER, PARAMETER :: n = 3, h_ne = 4, h_all = 5
   INTEGER, PARAMETER :: len_integer = 2 * n + 3 + h_all, len_real = h_all
   INTEGER, PARAMETER :: nflag = 2, st_flag = 2, st_ptr = st_flag + n
   INTEGER, PARAMETER :: st_row = st_ptr + n + 1, st_val = 0
   INTEGER, ALLOCATABLE, DIMENSION( : ) :: B_stat
   EXTERNAL :: HPROD
! start problem data
   ALLOCATE( p%G( n ), p%X_l( n ), p%X_u( n ) )
   ALLOCATE( p%X( n ), p%Z( n ) )
   ALLOCATE( userdata%integer( len_integer ), userdata%real( len_real ) )
   ALLOCATE( B_stat( n ) )
   p%new_problem_structure = .TRUE.           ! new structure
   p%n = n ; p%f = 1.0_wp                     ! dimensions & objective constant
   p%G = (/ 0.0_wp, 2.0_wp, 1.0_wp /)         ! objective gradient
   p%X_l = (/ - 1.0_wp, - infinity, 0.0_wp /) ! variable lower bound
   p%X_u = (/ infinity, 1.0_wp, 2.0_wp /)     ! variable upper bound
   p%X = 0.0_wp ; p%Z = 0.0_wp ! start from zero
! whole Hessian by rows for efficient matrix-vector products
   userdata%integer( st_ptr + 1 : st_ptr + n + 1 ) = (/ 1, 3, 5, 6 /)
   userdata%integer( st_row + 1 : st_row + h_all ) = (/ 1, 2, 1, 2, 3 /)
   userdata%real( st_val + 1 : st_val + h_all )                                &
     = (/ 1.0_wp, 1.0_wp, 1.0_wp, 2.0_wp, 3.0_wp /)
! problem data complete   
   CALL BQP_initialize( data, control, inform ) ! Initialize control parameters
   control%infinity = infinity                ! Set infinity
!  control%print_level = 1                    ! print one line/iteration
   control%maxit = 40                         ! limit the # iterations
!  control%print_gap = 100                    ! print every 100 terations
!  control%exact_gcp = .FALSE.
   userdata%integer( 1 ) = n
   userdata%integer( nflag ) = 0
   userdata%integer( st_flag + 1 : st_flag + n ) = 0
   inform%status = 1
   CALL BQP_solve( p,  B_stat, data, control, inform, userdata,                &
                   eval_HPROD = HPROD )
   IF ( inform%status == 0 ) THEN             !  Successful return
     WRITE( 6, "( ' BQP: ', I0, ' iterations  ', /,                            &
    &     ' Optimal objective value =',                                        &
    &       ES12.4, /, ' Optimal solution = ', ( 5ES12.4 ) )" )                &
     inform%iter, inform%obj, p%X
   ELSE
     WRITE( 6, "( ' BQP_solve exit status = ', I6 ) " ) inform%status
   END IF
   CALL BQP_terminate( data, control, inform )  !  delete workspace
   DEALLOCATE( p%G, p%X, p%X_l, p%X_u, p%Z, B_stat )
   DEALLOCATE( userdata%integer, userdata%real )
   END PROGRAM GALAHAD_BQP_THIRD_EXAMPLE

     SUBROUTINE HPROD( status, userdata, V, PROD, NZ_v, nz_v_start, nz_v_end,  &
                       NZ_prod, nz_prod_end )
! compute the matrix-vector product H * v
     USE GALAHAD_USERDATA_double
     INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
     INTEGER, INTENT( OUT ) :: status
     TYPE ( GALAHAD_userdata_type ), INTENT( INOUT ) :: userdata
     REAL ( KIND = wp ), DIMENSION( : ), INTENT( IN ) :: V
     REAL ( KIND = wp ), DIMENSION( : ), INTENT( OUT ) :: PROD
     INTEGER, OPTIONAL, INTENT( IN ) :: nz_v_start, nz_v_end
     INTEGER, OPTIONAL, INTENT( INOUT ) :: nz_prod_end
     INTEGER, DIMENSION( : ), OPTIONAL, INTENT( INOUT ) :: NZ_v
     INTEGER, DIMENSION( : ), OPTIONAL, INTENT( INOUT ) :: NZ_prod
     INTEGER :: i, j, k, l, n, nflag, st_flag, st_ptr, st_row, st_val
     REAL ( KIND = wp ) :: v_j
     n = userdata%integer( 1 )
     nflag = 2
     st_flag = 2
     st_ptr = st_flag + n
     st_row = st_ptr + n + 1
     st_val = 0
! compute H * v for very sparse v and record nonzeros
     IF ( PRESENT( NZ_prod ) .AND. PRESENT( nz_prod_end ) ) THEN
       userdata%integer( nflag ) = userdata%integer( nflag ) + 1
       nz_prod = 0
       DO l = nz_v_start, nz_v_end
         j = NZ_v( l ) ; v_j = V( j )
         DO k = userdata%integer( st_ptr + j ),                                &
                userdata%integer( st_ptr + j + 1 ) - 1
           i = userdata%integer( st_row + k )
           IF ( userdata%integer( st_flag + i ) <                              &
                userdata%integer( nflag ) ) THEN
             userdata%integer( st_flag + i ) = userdata%integer( nflag )
             PROD( i ) = userdata%real( st_val + k ) * v_j
             nz_prod_end = nz_prod_end + 1
             NZ_prod( nz_prod_end ) = i
           ELSE
             PROD( i ) = PROD( i ) + userdata%real( st_val + k ) * v_j
           END IF
         END DO
       END DO
! compute H * v for sparse v
     ELSE IF ( PRESENT( NZ_v ) .AND. PRESENT( nz_v_start ) .AND.               &
               PRESENT( nz_v_end ) ) THEN 
       PROD = 0.0_wp
       DO l = nz_v_start, nz_v_end
         j = NZ_v( l ) ; v_j = V( j )
         DO k = userdata%integer( st_ptr + j ),                                &
                userdata%integer( st_ptr + j + 1 ) - 1
           i = userdata%integer( st_row + k )
           PROD( i ) = PROD( i ) + userdata%real( st_val + k ) * v_j
         END DO
       END DO
! compute H * v
     ELSE
       PROD = 0.0_wp
       DO j = 1, n
         v_j = V( j )
         DO k = userdata%integer( st_ptr + j ),                                &
                userdata%integer( st_ptr + j + 1 ) - 1
           i = userdata%integer( st_row + k )
           PROD( i ) = PROD( i ) + userdata%real( st_val + k ) * v_j
         END DO
       END DO
     END IF
     status = 0
     END SUBROUTINE HPROD
