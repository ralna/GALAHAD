! THIS VERSION: GALAHAD 2.5 - 31/05/2021 AT 09:30 GMT.

!-*-*-*-*-*-*-*-*-*-  G A L A H A D   U S E _ S Q P  -*-*-*-*-*-*-*-*-*-*-

!  Nick Gould, for GALAHAD productions
!  Copyright reserved
!  Started: March 8th 2006

   MODULE GALAHAD_USESQP_double

     USE GALAHAD_CUTEST_FUNCTIONS_double
     USE GALAHAD_COPYRIGHT
     USE GALAHAD_SMT_double
     USE GALAHAD_SBLS_double
     USE GALAHAD_SPACE_double
     USE GALAHAD_QPC_double
     USE GALAHAD_NLPT_double, ONLY: NLPT_problem_type
     USE GALAHAD_SYMBOLS

     IMPLICIT NONE
     PRIVATE
     PUBLIC :: USE_SQP_DPR

!  Set precision

     INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )

!  Set other parameters

     REAL ( KIND = wp ), PARAMETER ::  zero = 0.0_wp
     REAL ( KIND = wp ), PARAMETER ::  one = 1.0_wp
     REAL ( KIND = wp ), PARAMETER ::  ten = 10.0_wp

     INTEGER, PARAMETER :: out = 6
     INTEGER, PARAMETER :: error = 6
     INTEGER, PARAMETER :: io_buffer = 11
!    INTEGER, PARAMETER :: out_sol = 23
     INTEGER, PARAMETER :: out_sol = 0
     INTEGER, PARAMETER :: it_max = 20
!    INTEGER, PARAMETER :: it_max = 2
     INTEGER :: sfiledevice = 62
     CHARACTER ( LEN = 30 ) :: sfilename = 'SQPSOL.d'
     REAL ( KIND = wp ), PARAMETER ::  pr_opt = ten ** ( - 12 )
     REAL ( KIND = wp ), PARAMETER ::  du_opt = ten ** ( - 12 )
     REAL ( KIND = wp ), PARAMETER ::  infinity = ten ** 19

   CONTAINS

!-*-*-*-*-*-*-*-*-*-*-  U S E _ S Q P   S U B R O U T I N E  -*-*-*-*-*-*-*-*-*-

     SUBROUTINE USE_SQP_DPR( input )

!  Dummy argument

     INTEGER, INTENT( IN ) :: input

!  Simple SQP method without linesearch

     INTEGER :: i, l, iter, iores, smt_stat
     INTEGER :: n, m, npm, J_ne, H_ne, J_len, H_len
     INTEGER :: status, alloc_status, cutest_status
     LOGICAL :: grlagf, filexx, inequality, use_merit
     CHARACTER ( len = 1 ) :: pert
 
     REAL ( KIND = wp ) :: obj, pr_feas, du_feas, dx, dy, alpha, merit, sigma
     REAL ( KIND = wp ) :: merit_trial
     REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: RHS, Y, X_trial, C_trial
     INTEGER, ALLOCATABLE, DIMENSION( : ) :: B_stat, C_stat

     TYPE ( NLPT_problem_type ) :: nlp
     TYPE ( SMT_type ) :: A, H, C
     TYPE ( SBLS_data_type ) :: SBLS_data
     TYPE ( SBLS_control_type ) :: SBLS_control
     TYPE ( SBLS_inform_type ) :: SBLS_inform
     TYPE ( QPT_problem_type ) :: prob
     TYPE ( QPC_data_type ) :: data
     TYPE ( QPC_control_type ) :: QPC_control        
     TYPE ( QPC_inform_type ) :: QPC_inform

!  Set copyright 

     IF ( out > 0 ) CALL COPYRIGHT( out, '2006' )

     CALL CUTEST_cdimen( cutest_status, input, nlp%n, nlp%m )
     IF ( cutest_status /= 0 ) GO TO 910
     n = nlp%n ; m = nlp%m ; npm = n + m

!  Allocate arrays

     CALL SPACE_resize_array( n, nlp%X, status, alloc_status )
     IF ( status /= 0 ) GO TO 990

     CALL SPACE_resize_array( n, nlp%X_l, status, alloc_status )
     IF ( status /= 0 ) GO TO 990

     CALL SPACE_resize_array( n, nlp%X_u, status, alloc_status )
     IF ( status /= 0 ) GO TO 990

     CALL SPACE_resize_array( m, nlp%Y, status, alloc_status )
     IF ( status /= 0 ) GO TO 990

     CALL SPACE_resize_array( n, nlp%Z, status, alloc_status )
     IF ( status /= 0 ) GO TO 990

     CALL SPACE_resize_array( n, nlp%G, status, alloc_status )
     IF ( status /= 0 ) GO TO 990

     CALL SPACE_resize_array( n, nlp%gL, status, alloc_status )
     IF ( status /= 0 ) GO TO 990

     CALL SPACE_resize_array( m, nlp%C, status, alloc_status )
     IF ( status /= 0 ) GO TO 990

     CALL SPACE_resize_array( m, nlp%C_l, status, alloc_status )
     IF ( status /= 0 ) GO TO 990

     CALL SPACE_resize_array( m, nlp%C_u, status, alloc_status )
     IF ( status /= 0 ) GO TO 990

     CALL SPACE_resize_array( m,nlp%EQUATION, status, alloc_status )
     IF ( status /= 0 ) GO TO 990

     CALL SPACE_resize_array( m, nlp%LINEAR, status, alloc_status )
     IF ( status /= 0 ) GO TO 990

     CALL SPACE_resize_array( m, nlp%CNAMES, status, alloc_status )
     IF ( status /= 0 ) GO TO 990

     CALL SPACE_resize_array( n, nlp%VNAMES, status, alloc_status )
     IF ( status /= 0 ) GO TO 990

     i = n
     CALL CUTEST_csetup( cutest_status, input, error, io_buffer, n, m,         &
                         nlp%X, nlp%X_l, nlp%X_u, nlp%Y, nlp%C_l, nlp%C_u,     &
                         nlp%EQUATION, nlp%LINEAR, 0, 0, 0 )
     IF ( cutest_status /= 0 ) GO TO 910
     inequality = .FALSE.
     DO i = 1, n
       IF ( nlp%X_l( i ) > - infinity .OR. nlp%X_u( i ) < infinity ) THEN
         IF ( nlp%X_l( i ) /= nlp%X_u( i ) ) THEN
           inequality = .TRUE.
           EXIT
         END IF
       END IF
     END DO
     DO i = 1, m
       IF ( nlp%C_l( i ) > - infinity .OR. nlp%C_u( i ) < infinity ) THEN
         IF ( nlp%C_l( i ) /= nlp%C_u( i ) ) THEN
           inequality = .TRUE.
           EXIT
         END IF
       END IF
     END DO

!  Obtain the names of the problem, its variables and general constraints

     CALL CUTEST_cnames( cutest_status, n, m,                                  &
                         nlp%pname, nlp%VNAMES, nlp%CNAMES )
     IF ( cutest_status /= 0 ) GO TO 910

!  Determine how many nonzeros are required to store the matrix of 
!  gradients of the objective function and constraints, when the matrix 
!  is stored in sparse format.

     CALL CUTEST_cdimsj( cutest_status, J_ne )
     IF ( cutest_status /= 0 ) GO TO 910

!  Determine how many nonzeros are required to store the Hessian matrix of the
!  Lagrangian, when the matrix is stored as a sparse matrix in "co-ordinate" 
!  format
 
     CALL CUTEST_cdimsh( cutest_status, H_ne )
     IF ( cutest_status /= 0 ) GO TO 910

     CALL SPACE_resize_array( J_ne, A%row, status, alloc_status )
     IF ( status /= 0 ) GO TO 990

     CALL SPACE_resize_array( J_ne, A%col, status, alloc_status )
     IF ( status /= 0 ) GO TO 990

     CALL SPACE_resize_array( J_ne, A%val, status, alloc_status )
     IF ( status /= 0 ) GO TO 990

     CALL SPACE_resize_array( H_ne, H%row, status, alloc_status )
     IF ( status /= 0 ) GO TO 990

     CALL SPACE_resize_array( H_ne, H%col, status, alloc_status )
     IF ( status /= 0 ) GO TO 990

     CALL SPACE_resize_array( H_ne, H%val, status, alloc_status )
     IF ( status /= 0 ) GO TO 990

! ------------------- inequality-constrained problem ---------------------

     IF ( inequality ) THEN

       CALL SPACE_resize_array( n, prob%G, status, alloc_status )
       IF ( status /= 0 ) GO TO 990

       CALL SPACE_resize_array( n, X_trial, status, alloc_status )
       IF ( status /= 0 ) GO TO 990

       CALL SPACE_resize_array( n, prob%X, status, alloc_status )
       IF ( status /= 0 ) GO TO 990

       CALL SPACE_resize_array( n, prob%X_l, status, alloc_status )
       IF ( status /= 0 ) GO TO 990

       CALL SPACE_resize_array( n, prob%X_u, status, alloc_status )
       IF ( status /= 0 ) GO TO 990

       CALL SPACE_resize_array( n, prob%Z, status, alloc_status )
       IF ( status /= 0 ) GO TO 990

       CALL SPACE_resize_array( m, C_trial, status, alloc_status )
       IF ( status /= 0 ) GO TO 990

       CALL SPACE_resize_array( m, prob%C, status, alloc_status )
       IF ( status /= 0 ) GO TO 990

       CALL SPACE_resize_array( m, prob%C_u, status, alloc_status )
       IF ( status /= 0 ) GO TO 990

       CALL SPACE_resize_array( m, prob%C_l, status, alloc_status )
       IF ( status /= 0 ) GO TO 990

       CALL SPACE_resize_array( m, prob%Y, status, alloc_status )
       IF ( status /= 0 ) GO TO 990

       CALL SPACE_resize_array( J_ne, prob%A%row, status, alloc_status )
       IF ( status /= 0 ) GO TO 990

       CALL SPACE_resize_array( J_ne, prob%A%col, status, alloc_status )
       IF ( status /= 0 ) GO TO 990

       CALL SPACE_resize_array( J_ne, prob%A%val, status, alloc_status )
       IF ( status /= 0 ) GO TO 990

       CALL SPACE_resize_array( H_ne, prob%H%row, status, alloc_status )
       IF ( status /= 0 ) GO TO 990

       CALL SPACE_resize_array( H_ne, prob%H%col, status, alloc_status )
       IF ( status /= 0 ) GO TO 990

       CALL SPACE_resize_array( H_ne, prob%H%val, status, alloc_status )
       IF ( status /= 0 ) GO TO 990

       CALL SPACE_resize_array( n, B_stat, status, alloc_status )
       IF ( status /= 0 ) GO TO 990

       CALL SPACE_resize_array( n, C_stat, status, alloc_status )
       IF ( status /= 0 ) GO TO 990

       prob%X = nlp%X
       prob%Z = zero ; nlp%Y = prob%Y
       prob%Y = zero ; nlp%Z = prob%Z
       prob%new_problem_structure = .TRUE.     
       prob%n = n
       prob%m = m
       CALL SMT_put( prob%H%type, 'COORDINATE', smt_stat )
       CALL SMT_put( prob%A%type, 'COORDINATE', smt_stat )

       CALL QPC_initialize( data, QPC_control, QPC_inform )
!      IF ( is_specfile )                                                      &
!        CALL QPC_read_specfile( QPC_control, input_specfile )

!  ---- Main SQP iteration -----

       WRITE( out, "( ' inequality problem: ', A )" ) nlp%pname

       WRITE( out, "( ' iter       pr_feas            du_feas      ',          &
      &               '    dx         dy           f' )" )

!  Evaluate the function and constraint values

       CALL CUTEST_cfn( cutest_status, n, m, nlp%X, nlp%f, nlp%C )
       IF ( cutest_status /= 0 ) GO TO 910

       pr_feas = MAX( MAXVAL( MAX( nlp%X_l - nlp%X, zero ) ),                  &
                      MAXVAL( MAX( nlp%X - nlp%X_u, zero ) ) )
       IF ( m > 0 ) pr_feas = MAX( MAXVAL( MAX( nlp%C_l - nlp%C, zero ) ),     &
                        MAXVAL( MAX( nlp%C - nlp%C_u, zero ) ), pr_feas )

       use_merit = .FALSE.
!      use_merit = .TRUE.
       IF ( use_merit ) THEN
         sigma = one
         merit = nlp%f + sigma * pr_feas
       END IF

       iter = 0
       DO

!        WRITE(6,"(A, /, ( 4ES22.14 ) )" ) ' xnew ',  nlp%X

         IF ( out_sol > 0 ) WRITE( out_sol, "( ' X', / ( 3ES24.16 ) )" ) nlp%X
         IF ( out_sol > 0 ) WRITE( out_sol, "( ' Y', / ( 3ES24.16 ) )" ) nlp%Y

!  Evaluate the Jacobian and Hessian

         grlagf = .FALSE. ; J_len = J_ne ; H_len = H_ne
         CALL CUTEST_csgrsh( cutest_status, n, m, nlp%X, - nlp%Y, grlagf,      &
                             J_ne, J_len, A%val, A%col, A%row,                 &
                             H%ne, H_len, H%val, H%row, H%col )
         IF ( cutest_status /= 0 ) GO TO 910

!  Untangle A: separate the gradient terms from the constraint Jacobian

         A%ne = 0 ; nlp%G( : n ) = zero
         DO l = 1, J_ne
           IF ( A%row( l ) == 0 ) THEN
             nlp%G( A%col( l ) ) = A%val( l )
           ELSE
             A%ne = A%ne + 1
             A%row( A%ne ) = A%row( l )
             A%col( A%ne ) = A%col( l )
             A%val( A%ne ) = A%val( l )
!            WRITE( out, " ( 2I8, ES12.4 )" ) A%row( A%ne ),                   &
!              A%col( A%ne ), A%val( A%ne ) 
           END IF
         END DO

!  Compute the gradient of the Lagrangian

!write(6,*) ' g ', nlp%G 
!write(6,*) ' z ', nlp%Z

         nlp%gL = nlp%G - nlp%Z
         DO l = 1, A%ne
           i = A%col( l )
           nlp%gL( i ) = nlp%gL( i ) - A%val( l ) * nlp%Y( A%row( l ) )
         END DO
         du_feas = MAXVAL( ABS( nlp%gL ) )

         IF ( iter == 0 ) THEN
           WRITE( out, "( I5, 1X, 2ES19.12,'     -          -     ', ES12.4)") &
             iter, pr_feas, du_feas, nlp%f
         ELSE
           WRITE( out, "( I5, 1X, 2ES19.12, 2ES11.4, ES12.4 )" )               &
            iter, pr_feas, du_feas, dx, dy, nlp%f
         END IF

         IF ( pr_feas < pr_opt .AND. du_feas < du_opt ) THEN
           status = GALAHAD_ok
           EXIT
         END IF

         iter = iter + 1
         IF ( iter > it_max ) THEN
           status = GALAHAD_error_max_iterations
           EXIT
         END IF

         prob%f = nlp%f
         prob%G = nlp%gL
         prob%X = zero ; prob%Y = zero ; prob%Z = zero
         prob%X_l = nlp%X_l - nlp%X ; prob%X_u = nlp%X_u - nlp%X
         prob%C_l = nlp%C_l - nlp%C ; prob%C_u = nlp%C_u - nlp%C

         prob%A%ne = A%ne
         prob%A%row = A%row ; prob%A%col = A%col ; prob%A%val = A%val
         prob%H%ne = H%ne
         prob%H%row = H%row ; prob%H%col = H%col ; prob%H%val = H%val

         C_stat = 0 ; B_stat = 0

!        QPC_control%print_level = 1
!        QPC_control%QPB_control%print_level = 1
!        QPC_control%QPB_control%maxit = 1
!        QPC_control%QPB_control%SBLS_control%preconditioner = 1
         CALL QPC_solve( prob, C_stat, B_stat, data, QPC_control, QPC_inform )

         IF ( QPC_inform% status < 0 ) THEN
           status = GALAHAD_error_qpc
           EXIT
         END IF

         dx = MAXVAL( ABS( prob%X ) )
         IF ( m > 0 ) THEN ; dy = MAXVAL( ABS( prob%Y ) )
           ELSE ; dy = zero ; END IF
 
         IF ( use_merit ) THEN
           sigma = MAX( sigma, 1.1 * MAXVAL( ABS( nlp%Z ) ) )
           IF ( m > 0 ) sigma = MAX( sigma, 1.1 * MAXVAL( ABS( nlp%Y ) ) )
           merit = nlp%f + sigma * pr_feas

!  Evaluate the function and constraint values

           alpha = one
           write(6, "( ' alpha, merit ', 2ES12.4 )" ) zero, merit
           DO 
             X_trial = nlp%X + alpha * prob%X
             CALL CUTEST_cfn( cutest_status,  n, m, X_trial, nlp%f, nlp%C )
             IF ( cutest_status /= 0 ) GO TO 910

             pr_feas = MAX( MAXVAL( MAX( nlp%X_l - nlp%X, zero ) ),            &
                            MAXVAL( MAX( nlp%X - nlp%X_u, zero ) ) )
             IF ( m > 0 ) pr_feas = MAX( MAXVAL( MAX( nlp%C_l - nlp%C, zero )),&
                              MAXVAL( MAX( nlp%C - nlp%C_u, zero ) ), pr_feas )

             merit_trial = nlp%f + sigma * pr_feas
             write(6, "( ' alpha, merit ', 2ES12.4 )" ) alpha, merit_trial
             IF ( merit_trial < merit ) THEN
               nlp%X = X_trial
               EXIT
             ELSE
               alpha = 0.5_wp * alpha
             END IF
           END DO
         ELSE
           nlp%X = nlp%X + prob%X
           CALL CUTEST_cfn( cutest_status,  n, m, nlp%X, nlp%f, nlp%C )
           IF ( cutest_status /= 0 ) GO TO 910
           pr_feas = MAX( MAXVAL( MAX( nlp%X_l - nlp%X, zero ) ),              &
                          MAXVAL( MAX( nlp%X - nlp%X_u, zero ) ) )
           IF ( m > 0 ) pr_feas = MAX( MAXVAL( MAX( nlp%C_l - nlp%C, zero ) ), &
                          MAXVAL( MAX( nlp%C - nlp%C_u, zero ) ), pr_feas )
         END IF
         nlp%Y = prob%Y
         nlp%Z = prob%Z

       END DO

! ------------------- equality-constrained problem ---------------------

     ELSE

!  Allocate more arrays

       CALL SPACE_resize_array( m, Y, status, alloc_status )
       IF ( status /= 0 ) GO TO 990

       CALL SPACE_resize_array( npm, RHS, status, alloc_status )
       IF ( status /= 0 ) GO TO 990

       CALL SMT_put( H%type, 'COORDINATE', smt_stat )
       CALL SMT_put( A%type, 'COORDINATE', smt_stat )
       CALL SMT_put( C%type, 'ZERO', smt_stat )

       CALL SBLS_initialize( SBLS_data, SBLS_control, SBLS_inform )
       SBLS_control%preconditioner = 2
       SBLS_control%factorization = 2
       SBLS_control%itref_max = 2

!      SBLS_control%print_level = 2

!  ---- Main SQP iteration -----

       WRITE( out, "( ' equality problem: ', A )" ) nlp%pname

       iter = 0
       WRITE( out, "( ' iter        pr_feas               du_feas       ',     &
      &               '      dx          dy' )" )

       pert = ' '

       DO

!        WRITE(6,"(A, /, ( 4ES22.14 ) )" ) ' xnew ',  nlp%X

         IF ( out_sol > 0 ) WRITE( out_sol, "( ' X', / ( 3ES24.16 ) )" ) nlp%X
         IF ( out_sol > 0 ) WRITE( out_sol, "( ' Y', / ( 3ES24.16 ) )" ) nlp%Y

!  Evaluate the function and constraint values

         CALL CUTEST_cfn( cutest_status,  n, m, nlp%X, obj, nlp%C )
         IF ( cutest_status /= 0 ) GO TO 910
         pr_feas = MAXVAL( ABS( nlp%C ) )
!        IF ( out_sol > 0 ) WRITE( out_sol, "( ' f ', ES24.16 )" ) obj

!write(6,*) ' c ', nlp%C

!  Evaluate the Jacobian and Hessian

         grlagf = .FALSE. ; J_len = J_ne ; H_len = H_ne
!        Y = - nlp%Y
         CALL CUTEST_csgrsh( cutest_status, n, m, nlp%X, - nlp%Y, grlagf,      &
                            J_ne, J_len, A%val, A%col, A%row,                  &
                            H%ne, H_len, H%val, H%row, H%col )
         IF ( cutest_status /= 0 ) GO TO 910

!write(6,*) ' H: row ', H%row(:H%ne)
!write(6,*) ' H: col ', H%col(:H%ne)
!write(6,*) ' H: val ', H%val(:H%ne)

!  Untangle A: separate the gradient terms from the constraint Jacobian

         A%ne = 0 ; nlp%G( : n ) = zero
         DO l = 1, J_ne
           IF ( A%row( l ) == 0 ) THEN
             nlp%G( A%col( l ) ) = A%val( l )
           ELSE
             A%ne = A%ne + 1
             A%row( A%ne ) = A%row( l )
             A%col( A%ne ) = A%col( l )
             A%val( A%ne ) = A%val( l )
!            WRITE( out, " ( 2I8, ES12.4 )" ) A%row( A%ne ),                   &
!              A%col( A%ne ), A%val( A%ne ) 
           END IF
         END DO

!write(6,*) ' A: row ', A%row(:A%ne)
!write(6,*) ' A: col ', A%col(:A%ne)
!write(6,*) ' A: val ', A%val(:A%ne)

!  Compute the gradient of the Lagrangian

         nlp%gL = nlp%G
         DO l = 1, A%ne
           i = A%col( l )
           nlp%gL( i ) = nlp%gL( i ) - A%val( l ) * nlp%Y( A%row( l ) )
         END DO
         du_feas = MAXVAL( ABS( nlp%gL ) )

         IF ( iter == 0 ) THEN
           WRITE( out, "( I5, A1, 2ES22.14, '       -           - ' )" )       &
             iter, pert, pr_feas, du_feas
         ELSE
           WRITE( out, "( I5, A1, 2ES22.14, 2ES12.4 )" )                       &
            iter, pert, pr_feas, du_feas, dx, dy
         END IF

         IF ( pr_feas < pr_opt .AND. du_feas < du_opt ) THEN
           status = GALAHAD_ok
           EXIT
         END IF

         iter = iter + 1
         IF ( iter > it_max ) THEN
           status = GALAHAD_error_max_iterations
           EXIT
         END IF

         pert = ' '


!  Form and factorize the KKT matrix

!        SBLS_control%out = 6
!        SBLS_control%print_level = 1

         CALL SBLS_form_and_factorize( n, m, H, A, C, SBLS_data,               &
                                       SBLS_control, SBLS_inform )

         IF ( SBLS_inform%status /= 0 ) THEN
           status = GALAHAD_error_factorization
           WRITE( out, "( ' factorization error: status = ', I0 )" )           &
             SBLS_inform%status
           EXIT
         END IF
         IF ( SBLS_inform%perturbed ) pert = 'm'

!  Find the Newton correction

         RHS( 1 : n ) = - nlp%gL
         RHS( n + 1 : npm ) = - nlp%C
!write(6,*) ' rhs ', RHS(:npm)
         CALL SBLS_solve( n, m, A, C, SBLS_data, SBLS_control, SBLS_inform,RHS)
!write(6,*) ' sol ', RHS(:npm)

         IF ( SBLS_inform%status /= 0 ) THEN
           status = GALAHAD_error_factorization
           WRITE( out, "( ' solve error: status = ', I0 )" ) SBLS_inform%status
           EXIT
         END IF

!  Update the solution estimate

         IF ( out_sol > 0 ) WRITE( out_sol, "( ' DX', / ( 3ES24.16 ) )" )      &
           RHS( 1 : n )
         IF ( out_sol > 0 ) WRITE( out_sol, "( ' DY', / ( 3ES24.16 ) )" )      &
           RHS( n + 1 : npm )

         dx = MAXVAL( ABS( RHS( 1 : n ) ) )
         dy = MAXVAL( ABS( RHS( n + 1 : npm ) ) )
         nlp%X = nlp%X + RHS( 1 : n )
         nlp%Y = nlp%Y - RHS( n + 1 : npm )

!  ---- End of main SQP iteration -----

       END DO
     END IF

!    WRITE( out, "( /, ' Solution ', /, ' X', / ( 3ES24.16 ) )" ) nlp%X
!    IF ( m > 0 ) WRITE( out, "( ' Y', / ( 3ES24.16 ) )" ) nlp%Y
     IF ( out_sol > 0 ) WRITE( out_sol, "( ' X', / ( 3ES24.16 ) )" ) nlp%X
     IF ( out_sol > 0 ) WRITE( out_sol, "( ' Y', / ( 3ES24.16 ) )" ) nlp%Y

!  Write out the solution

     INQUIRE( FILE = sfilename, EXIST = filexx )
     IF ( filexx ) THEN
        OPEN( sfiledevice, FILE = sfilename, FORM = 'FORMATTED',               &
            STATUS = 'OLD', IOSTAT = iores )
     ELSE
        OPEN( sfiledevice, FILE = sfilename, FORM = 'FORMATTED',               &
            STATUS = 'NEW', IOSTAT = iores )
     END IF
     IF ( iores /= 0 ) THEN 
       WRITE( out, "( ' IOSTAT = ', I6, ' when opening file ', A9 )" )         &
         iores, sfilename
       RETURN
     END IF

     WRITE( sfiledevice, "( '*   SQP solution for problem name: ', A8 )" )     &
       nlp%pname
     WRITE( sfiledevice, "( /, '*   variables ', / )" )
     DO i = 1, n
       WRITE( sfiledevice, "( '    Solution  ', A10, ES12.5 )" )               &
         nlp%VNAMES( i ), nlp%X( i )
     END DO
     WRITE( sfiledevice, "( /, '*   Lagrange multipliers ', / )" )
     DO i = 1, m
        WRITE( sfiledevice, "( ' M  Solution  ', A10, ES12.5 )" )              &
          nlp%CNAMES( i ), nlp%Y( i )
     END DO
     WRITE( sfiledevice, "( /, ' XL Solution  ', 10X, ES12.5 )" ) obj
     CLOSE( sfiledevice ) 

     IF ( status == GALAHAD_ok ) THEN
       WRITE( 6, "( /, ' SQP successful termination, objective =', ES12.4 )" ) &
         obj
     ELSE
       WRITE( 6, "( /, ' SQP unsuccessful termination, status = ', I0 )" )     &
         status
     END IF
     CALL CUTEST_cterminate( cutest_status )
     RETURN

 910 CONTINUE
     WRITE( out, "( ' CUTEst error, status = ', i0, ', stopping' )")           &
        cutest_status
     status = GALAHAD_error_evaluation
     RETURN

 990 CONTINUE
     WRITE( out, "( ' allocation error ', I0, ' status ', I0 )" )              &
       status, alloc_status
     status = GALAHAD_error_allocate
     RETURN

     END SUBROUTINE USE_SQP_DPR

   END MODULE GALAHAD_USESQP_double
