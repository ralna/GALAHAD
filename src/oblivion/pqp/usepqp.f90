! THIS VERSION: GALAHAD 3.3 - 20/05/2021 AT 11:00 GMT.

!-*-*-*-*-*-*-*-*-  G A L A H A D   U S E P Q P  _ m a i n  *-*-*-*-*-*-*-*-

!  Nick Gould, for GALAHAD productions
!  Copyright reserved

    MODULE GALAHAD_USEPQP_double

!  CUTEst/AMPL interface to PQP, an algorithm for solving parameteric 
!  quadratic programs.

      USE CUTEst_interface_double
!NOT95USE GALAHAD_CPU_time
      USE GALAHAD_RAND_double
      USE GALAHAD_SYMBOLS
      USE GALAHAD_QPT_double
      USE GALAHAD_PQP_double
      USE GALAHAD_SORT_double, only: SORT_reorder_by_rows
      USE GALAHAD_SPECFILE_double 
      USE GALAHAD_COPYRIGHT
      USE GALAHAD_SCALING_double
      USE GALAHAD_SYMBOLS,                                                     &
          ACTIVE                => GALAHAD_ACTIVE,                             &
          GENERAL               => GALAHAD_GENERAL,                            &
          ALL_ZEROS             => GALAHAD_ALL_ZEROS

      IMPLICIT NONE

      PRIVATE
      PUBLIC :: USE_PQP

    CONTAINS

!-*-*-*-*-*-*-*-*-*-   U S E _ P Q P    S U B R O U T I N E   -*-*-*-*-*-*-*-

      SUBROUTINE USE_PQP( input )

!  ----------------------------------------------------------------------
!
!  Solve the parametric quadratic program from CUTEst
!
!   QP(theta):  minimize   1/2 x(T) H x + g(T) x + f + theta dg(T) x
!               subject to c_l + theta dc_l <= A x <= c_u + theta dc_u
!               and        x_l + theta dx_l <=  x  <= x_u + theta dx_u
!
!  for all 0 <= theta <= theta_max. 
!
!  The vectors dg, dc_l, dc_u, dx_l, dx_u are randomly generated in [-1.1]
!
!  ----------------------------------------------------------------------

!  Dummy argument

      INTEGER, INTENT( IN ) :: input

!  Parameters

      INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
!     INTEGER, PARAMETER :: n_pass = 1
!     INTEGER, PARAMETER :: n_pass = 2
      INTEGER, PARAMETER :: n_pass = 3
      REAL ( KIND = wp ), PARAMETER :: zero = 0.0_wp
      REAL ( KIND = wp ), PARAMETER :: half = 0.5_wp
      REAL ( KIND = wp ), PARAMETER :: one = 1.0_wp
      REAL ( KIND = wp ), PARAMETER :: two = 2.0_wp
      REAL ( KIND = wp ), PARAMETER :: ten = 10.0_wp
      REAL ( KIND = wp ), PARAMETER :: infinity = ten ** 19

!  Scalars

      INTEGER :: n, m, ir, ic, ifail, la, lh, liw, iores, a_ne, h_ne
!     INTEGER :: np1, npm, nmods, newton
      INTEGER :: i, j, l, neh, nea, factorization_integer, factorization_real
      INTEGER :: status, mfixed, mdegen, iter, nfacts, nfixed, ndegen, mequal
      INTEGER :: alloc_stat, cutest_status, pass
      INTEGER :: miss_ident, s_iter( n_pass ), s_status( n_pass )
      REAL :: time, timeo, times, timet, s_timet( n_pass )
      REAL ( KIND = wp ) :: objf, qfval, stopr, rho_g, rho_b, randm, theta_max
      REAL ( KIND = wp ) :: s_qfval( n_pass ), theta_u, theta_end
      REAL ( KIND = wp ) :: cl, cu, xl, xu
      LOGICAL :: filexx, is_specfile, printo, printe, random = .FALSE.
!     LOGICAL :: print_solution = .FALSE.
      LOGICAL :: print_solution = .TRUE.
      TYPE ( RAND_seed ) :: seed

!  Specfile characteristics

      INTEGER, PARAMETER :: input_specfile = 34
      INTEGER, PARAMETER :: lspec = 22
      CHARACTER ( LEN = 16 ) :: specname = 'RUNQPA'
      TYPE ( SPECFILE_item_type ), DIMENSION( lspec ) :: spec
      CHARACTER ( LEN = 16 ) :: runspec = 'RUNQPA.SPC'

!  Default values for specfile-defined parameters

      INTEGER :: scale = 0
      INTEGER :: dfiledevice = 26
      INTEGER :: ifiledevice = 51
      INTEGER :: rfiledevice = 47
      INTEGER :: sfiledevice = 62
      LOGICAL :: write_problem_data   = .FALSE.
      LOGICAL :: write_initial_sif    = .FALSE.
      LOGICAL :: write_solution       = .FALSE.
      LOGICAL :: write_result_summary = .FALSE.
      CHARACTER ( LEN = 30 ) :: dfilename = 'QPA.data'
      CHARACTER ( LEN = 30 ) :: ifilename = 'initial.sif'
      CHARACTER ( LEN = 30 ) :: rfilename = 'PARAM_RES.d'
      CHARACTER ( LEN = 30 ) :: sfilename = 'PARAM_SOL.d'
      LOGICAL :: do_solve = .TRUE.
      LOGICAL :: fulsol = .FALSE. 
      REAL ( KIND = wp ) :: pert = ten ** ( - 8 )
      INTEGER :: initial_seed = 2345671

!  Output file characteristics

      INTEGER, PARAMETER :: out  = 6
      INTEGER, PARAMETER :: io_buffer = 11
      INTEGER :: errout = 6
      CHARACTER ( LEN =  5 ) :: state
      CHARACTER ( LEN =  9 ) :: solv
      CHARACTER ( LEN = 10 ) :: pname

!  Arrays

      CHARACTER ( LEN = 20 ) :: action
      TYPE ( SCALING_control_type ) :: SCALING_control
      TYPE ( PQP_interval_type ) :: PQP_interval
      TYPE ( QPT_problem_type ) :: prob
      TYPE ( PQP_data_type ) :: PQP_data
      TYPE ( PQP_control_type ) :: PQP_control        
      TYPE ( PQP_inform_type ) :: PQP_inform

!  Allocatable arrays

      CHARACTER ( LEN = 10 ), ALLOCATABLE, DIMENSION( : ) :: VNAME, CNAME
      REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: SH, SA, X0, C, Y
      LOGICAL, ALLOCATABLE, DIMENSION( : ) :: EQUATN, LINEAR
      INTEGER, ALLOCATABLE, DIMENSION( : ) :: IW, C_stat, B_stat
      INTEGER, ALLOCATABLE, DIMENSION( : ) :: C_stat_old, B_stat_old

      CALL CPU_TIME( time )

!  Determine the number of variables and constraints

      CALL CUTEST_cdimen( cutest_status, input, n, m )
      IF ( cutest_status /= 0 ) GO TO 910

!  Allocate suitable arrays

      ALLOCATE( prob%X( n ), prob%X_l( n ), prob%X_u( n ), prob%G( n ),        &
                VNAME( n ), B_stat( n ), prob%DG( n ), prob%DX_l( n ),         &
                prob%DX_u( n ), B_stat_old( n ), STAT = alloc_stat )
      IF ( alloc_stat /= 0 ) THEN
        WRITE( out, 2150 ) 'X', alloc_stat ; STOP
      END IF

      ALLOCATE( prob%C_l( m ), prob%C_u( m ), Y( m ), prob%Y( m ),             &
                CNAME( m ), EQUATN( m ), LINEAR( m ), prob%DC_l( m ),          &
                prob%DC_u( m ), C_stat( m ), C_stat_old( m ),                  &
                STAT = alloc_stat )
      IF ( alloc_stat /= 0 ) THEN
        WRITE( out, 2150 ) 'C', alloc_stat ; STOP
      END IF

!  ------------------- set up problem ----------------------

!  Set up the data structures necessary to hold the group partially
!  separable function.

      CALL CUTEST_csetup( cutest_status, input, out, io_buffer,                &
                          n, m, prob%X, prob%X_l, prob%X_u,                    &
                          Y, prob%C_l, prob%C_u, EQUATN, LINEAR, 0, 0, 0 )
      IF ( cutest_status /= 0 ) GO TO 910
      DEALLOCATE( LINEAR )

!  Allocate derived types

      ALLOCATE( X0( n ), STAT = alloc_stat )
      IF ( alloc_stat /= 0 ) THEN
        WRITE( out, 2150 ) 'X0', alloc_stat
        STOP
      END IF

      ALLOCATE( prob%C( m ), C( m ), STAT = alloc_stat )
      IF ( alloc_stat /= 0 ) THEN
        WRITE( out, 2150 ) 'C', alloc_stat
        STOP
      END IF

!  Determine the names of the problem, variables and constraints.

      CALL CUTEST_cnames( cutest_status, n, m, pname, VNAME, CNAME )
      IF ( cutest_status /= 0 ) GO TO 910
      WRITE( out, 2020 ) pname 

!  Set up the initial estimate of the solution and
!  right-hand-side of the Kuhn-Tucker system.

!  Determine the constant terms for the problem functions.

      prob%X( : n ) = MIN( prob%X_u( : n ),                                    &
                           MAX( prob%X_l( : n ), prob%X( : n ) ) )

!  Set X0 to zero to determine the constant terms for the problem functions

      X0 = zero 

!  Evaluate the constant terms of the objective (objf) and constraint 
!  functions (C)

      CALL CUTEST_cfn( cutest_status, n, m, X0, objf, C( : m ) )
      IF ( cutest_status /= 0 ) GO TO 910
      DO i = 1, m 
        IF ( EQUATN( i ) ) THEN 
          prob%C_l( i ) = prob%C_l( i ) - C( i )
          prob%C_u( i ) = prob%C_l( i )
        ELSE
          prob%C_l( i ) = prob%C_l( i ) - C( i )
          prob%C_u( i ) = prob%C_u( i ) - C( i )
        END IF 
      END DO

!  Determine the number of nonzeros in the Jaccobian

      CALL CUTEST_cdimsj( cutest_status, la )
      IF ( cutest_status /= 0 ) GO TO 910
      la = MAX( la, 1 )

!  Allocate arrays to hold the Jacobian

      ALLOCATE( prob%A%row( la ), prob%A%col( la ), prob%A%val( la ),          &
                STAT = alloc_stat )
      IF ( alloc_stat /= 0 ) THEN
        WRITE( out, 2150 ) 'A', alloc_stat ; STOP
      END IF

!  Evaluate the linear terms of the constraint functions

      CALL CUTEST_csgr( cutest_status, n, m, X0, Y, .FALSE.,                   &
                        nea, la, prob%A%val, prob%A%col, prob%A%row )
      IF ( cutest_status /= 0 ) GO TO 910
!     DEALLOCATE( X0 )
      X0( : n ) = prob%X( : n )
      
!  Exclude zeros; set the linear term for the objective function

      A_ne = 0
      prob%G( : n )      = zero
      prob%gradient_kind = ALL_ZEROS
      DO i = 1, nea
        IF ( prob%A%val( i ) /= zero ) THEN
          IF ( prob%A%row( i ) > 0 ) THEN
            A_ne = A_ne + 1
            prob%A%row( A_ne ) = prob%A%row( i ) 
            prob%A%col( A_ne ) = prob%A%col( i )
            prob%A%val( A_ne ) = prob%A%val( i )
          ELSE
            prob%G( prob%A%col( i ) ) = prob%A%val( i )
            prob%gradient_kind        = GENERAL
          END IF  
        END IF
      END DO

!  Determine the number of nonzeros in the Hessian

      CALL CUTEST_cdimsh( cutest_status, lh )
      IF ( cutest_status /= 0 ) GO TO 910

!  Allocate arrays to hold the Hessian

      ALLOCATE( prob%H%row( lh ), prob%H%col( lh ), prob%H%val( lh ),          &
                STAT = alloc_stat )
      IF ( alloc_stat /= 0 ) THEN
        WRITE( out, "( ' nea = ', i8, ' la   = ', i8 )" ) nea, la
        WRITE( out, 2150 ) 'H', alloc_stat
        STOP
      END IF

!  Evaluate the Hessian of the Lagrangian function at the initial point.

      CALL CUTEST_csh( cutest_status, n, m, prob%X, Y,                         &
                       neh, lh, prob%H%val, prob%H%row, prob%H%col )
      IF ( cutest_status /= 0 ) GO TO 910
      WRITE( out, "( ' nea = ', i8, ' la   = ', i8,                            &
     &               ' neh  = ', i8, ' lh   = ', i8 )" ) nea, la, neh, lh

!  Remove Hessian out of range

      H_ne = 0
      DO l = 1, neh    
        i = prob%H%row( l ) ; j = prob%H%col( l )
        IF ( i < 1 .OR. i > n .OR. j < 1 .OR. j > n ) CYCLE
        H_ne = H_ne + 1 ; prob%H%val( H_ne ) = prob%H%val( l )
        IF ( i >= j ) THEN
          prob%H%row( H_ne ) = i
          prob%H%col( H_ne ) = j
        ELSE
          prob%H%row( H_ne ) = j
          prob%H%col( H_ne ) = i
        END IF
      END DO

!  Allocate and initialize dual variables.

      ALLOCATE( prob%Z( n ), STAT = alloc_stat )
      IF ( alloc_stat /= 0 ) THEN
        WRITE( out, 2150 ) 'Z', alloc_stat
        STOP
      END IF

      liw = MAX( m, n ) + 1
      ALLOCATE( prob%A%ptr( m + 1 ), prob%H%ptr( n + 1 ) )
      ALLOCATE( IW( liw ) )

!  Transform A to row storage format

      IF ( A_ne /= 0 ) THEN
        CALL SORT_reorder_by_rows( m, n, A_ne, prob%A%row, prob%A%col, A_ne,   &
                                   prob%A%val, prob%A%ptr, m + 1, IW, liw,     &
                                   out, out, i )
      ELSE
        prob%A%ptr = 0
      END IF

!  Same for H

      IF ( H_ne /= 0 ) THEN
        CALL SORT_reorder_by_rows( n, n, H_ne, prob%H%row, prob%H%col, H_ne,   &
                                   prob%H%val, prob%H%ptr, n + 1, IW, liw,     &
                                   out, out, i )
      ELSE
        prob%H%ptr = 0
      END IF

!  Deallocate arrays holding matrix row indices

      DEALLOCATE( prob%A%row, prob%H%row, IW )
      ALLOCATE( prob%A%row( 0 ), prob%H%row( 0 ) )

!  Store the problem dimensions

      prob%n = n
      prob%m = m
      prob%A%ne = - 1
      prob%H%ne = - 1
      prob%f = objf ; rho_g = 2 * m ; rho_b = 2 * n

!  ------------------- problem set-up complete ----------------------

      CALL CPU_TIME( times )
      times = times - time

!  ------------------ Open the specfile for runqpa -----------------

      INQUIRE( FILE = runspec, EXIST = is_specfile )
      IF ( is_specfile ) THEN
        OPEN( input_specfile, FILE = runspec, FORM = 'FORMATTED',              &
              STATUS = 'OLD' )

!   Define the keywords

        spec( 1 )%keyword = 'write-problem-data'
        spec( 2 )%keyword = 'problem-data-file-name'
        spec( 3 )%keyword = 'problem-data-file-device'
        spec( 4 )%keyword = 'write-initial-sif'
        spec( 5 )%keyword = 'initial-sif-file-name'
        spec( 6 )%keyword = 'initial-sif-file-device'
        spec( 7 )%keyword = 'initial-rho-g'
        spec( 8 )%keyword = 'initial-rho-b'
        spec( 9 )%keyword = 'scale-problem'
        spec( 14 )%keyword = 'solve-problem'
        spec( 15 )%keyword = 'print-full-solution'
        spec( 16 )%keyword = 'write-solution'
        spec( 17 )%keyword = 'solution-file-name'
        spec( 18 )%keyword = 'solution-file-device'
        spec( 19 )%keyword = 'write-result-summary'
        spec( 20 )%keyword = 'result-summary-file-name'
        spec( 21 )%keyword = 'result-summary-file-device'
        spec( 22 )%keyword = 'perturbation'

!   Read the specfile

        CALL SPECFILE_read( input_specfile, specname, spec, lspec, errout )

!   Interpret the result

        CALL SPECFILE_assign_logical( spec( 1 ), write_problem_data, errout )
        CALL SPECFILE_assign_string ( spec( 2 ), dfilename, errout )
        CALL SPECFILE_assign_integer( spec( 3 ), dfiledevice, errout )
        CALL SPECFILE_assign_logical( spec( 4 ), write_initial_sif, errout )
        CALL SPECFILE_assign_string ( spec( 5 ), ifilename, errout )
        CALL SPECFILE_assign_integer( spec( 6 ), ifiledevice, errout )
        CALL SPECFILE_assign_real( spec( 7 ), rho_g, errout )
        CALL SPECFILE_assign_real( spec( 8 ), rho_b, errout )
        CALL SPECFILE_assign_integer( spec( 9 ), scale, errout )
        CALL SPECFILE_assign_logical( spec( 14 ), do_solve, errout )
        CALL SPECFILE_assign_logical( spec( 15 ), fulsol, errout )
        CALL SPECFILE_assign_logical( spec( 16 ), write_solution, errout )
        CALL SPECFILE_assign_string ( spec( 17 ), sfilename, errout )
        CALL SPECFILE_assign_integer( spec( 18 ), sfiledevice, errout )
        CALL SPECFILE_assign_logical( spec( 19 ), write_result_summary, errout )
        CALL SPECFILE_assign_string ( spec( 20 ), rfilename, errout )
        CALL SPECFILE_assign_integer( spec( 21 ), rfiledevice, errout )
        CALL SPECFILE_assign_real( spec( 22 ), pert, errout )
      END IF

!  If required, print out the (raw) problem data

      IF ( write_problem_data ) THEN
        INQUIRE( FILE = dfilename, EXIST = filexx )
        IF ( filexx ) THEN
           OPEN( dfiledevice, FILE = dfilename, FORM = 'FORMATTED',            &
                 STATUS = 'OLD', IOSTAT = iores )
        ELSE
           OPEN( dfiledevice, FILE = dfilename, FORM = 'FORMATTED',            &
                  STATUS = 'NEW', IOSTAT = iores )
        END IF
        IF ( iores /= 0 ) THEN 
          write( out, 2160 ) iores, dfilename
          STOP
        END IF

        WRITE( dfiledevice, "( 'n, m = ', 2I6, ' obj = ', ES12.4 )" )          &
          n, m, prob%f
        WRITE( dfiledevice, "( ' g ', /, ( 5ES12.4 ) )" ) prob%G( : n )
        WRITE( dfiledevice, "( ' x_l ', /, ( 5ES12.4 ) )" ) prob%X_l( : n )
        WRITE( dfiledevice, "( ' x_u ', /, ( 5ES12.4 ) )" ) prob%X_u( : n )
        WRITE( dfiledevice, "( ' c_l ', /, ( 5ES12.4 ) )" ) prob%C_l( : m )
        WRITE( dfiledevice, "( ' c_u ', /, ( 5ES12.4 ) )" ) prob%C_u( : m )
        WRITE( dfiledevice, "( ' A_ptr ', /, ( 10I6 ) )" ) prob%A%ptr( : m + 1 )
        WRITE( dfiledevice, "( ' A_col ', /, ( 10I6 ) )" ) prob%A%col( : A_ne )
        WRITE( dfiledevice, "( ' A_val ', /, ( 5ES12.4 ) )" )                  &
          prob%A%val( : A_ne )
        WRITE( dfiledevice, "( ' H_ptr ', /, ( 10I6 ) )" ) prob%H%ptr( : n + 1 )
        WRITE( dfiledevice, "( ' H_col ', /, ( 10I6 ) )" ) prob%H%col( : H_ne )
        WRITE( dfiledevice, "( ' H_val ', /, ( 5ES12.4 ) )" )                  &
          prob%H%val( : H_ne )

        CLOSE( dfiledevice )
      END IF

!  If required, append results to a file

      IF ( write_result_summary ) THEN
        INQUIRE( FILE = rfilename, EXIST = filexx )
        IF ( filexx ) THEN
           OPEN( rfiledevice, FILE = rfilename, FORM = 'FORMATTED',            &
                 STATUS = 'OLD', POSITION = 'APPEND', IOSTAT = iores )
        ELSE
           OPEN( rfiledevice, FILE = rfilename, FORM = 'FORMATTED',            &
                 STATUS = 'NEW', IOSTAT = iores )
        END IF
        IF ( iores /= 0 ) THEN 
          write( out, 2160 ) iores, rfilename
          STOP
        END IF
        WRITE( rfiledevice, 2180 ) pname
      END IF

!  =========
!  Main loop
!  =========

      theta_max = zero
      DO pass = 1, n_pass

        s_iter( pass ) = - 1
        s_status( pass ) = - 1
        s_qfval( pass ) = infinity
        s_timet( pass ) = 1800.0

        prob%new_problem_structure = .TRUE.
        stopr = ten ** ( - 6 )
 
!  Set all default values, and override defaults if requested

        CALL PQP_initialize( PQP_interval, PQP_data,            &
                                  PQP_control )
        IF ( is_specfile )                                                     &
          CALL PQP_read_specfile( PQP_control, input_specfile )
        PQP_control%solve_qp = .TRUE.

!  Add the parametric part on the second pass
       
        IF ( pass == 2 ) THEN
!       IF ( pass > 1 ) THEN
          IF ( random ) THEN
            CALL RAND_initialize( seed )
            CALL RAND_set_seed( seed, initial_seed )

!  Parametric gradient

            DO i = 1, prob%n
               CALL RAND_random_real( seed, .FALSE., randm )
               prob%DG( i ) = randm
            END DO

!  Parametric simple bounds

            DO i = 1, prob%n
              IF ( prob%X_l( i ) == prob%X_u( i ) )  THEN
                CALL RAND_random_real( seed, .FALSE., randm )
                prob%DX_l( i ) = randm
                prob%DX_u( i ) = randm
              ELSE
                IF ( prob%X_l( i ) >= - infinity ) THEN
                  CALL RAND_random_real( seed, .FALSE., randm )
                  prob%DX_l( i ) = randm
                ELSE
                  prob%DX_l( i ) = zero
                END IF
                IF ( prob%X_u( i ) <= infinity ) THEN
                  CALL RAND_random_real( seed, .FALSE., randm )
                  prob%DX_u( i ) = randm
                ELSE
                  prob%DX_u( i ) = zero
                END IF
              END IF
            END DO

!  Parametric constraint bounds

            DO i = 1, prob%m
              IF ( prob%C_l( i ) == prob%C_u( i ) ) THEN
                CALL RAND_random_real( seed, .FALSE., randm )
                prob%DC_l( i ) = randm
                prob%DC_u( i ) = randm
              ELSE
                IF ( prob%C_l( i ) >= - infinity ) THEN
                  CALL RAND_random_real( seed, .FALSE., randm )
                  prob%DC_l( i ) = randm
                ELSE
                  prob%DC_l( i ) = zero
                END IF
                IF ( prob%X_u( i ) <= infinity ) THEN
                  CALL RAND_random_real( seed, .FALSE., randm )
                  prob%DC_u( i ) = randm
                ELSE
                  prob%DC_u( i ) = zero
                END IF
              END IF
            END DO
          ELSE
!  Parametric gradient

            DO i = 1, prob%n
               prob%DG( i ) = zero
            END DO

!  Parametric simple bounds

            DO i = 1, prob%n
              IF ( prob%X_l( i ) == prob%X_u( i ) )  THEN
                CALL RAND_random_real( seed, .FALSE., randm )
                prob%DX_l( i ) = one
                prob%DX_u( i ) = -one
              ELSE
                IF ( prob%X_l( i ) >= - infinity ) THEN
                  prob%DX_l( i ) = one
                ELSE
                  prob%DX_l( i ) = zero
                END IF
                IF ( prob%X_u( i ) <= infinity ) THEN
                  CALL RAND_random_real( seed, .FALSE., randm )
                  prob%DX_u( i ) = - one
                ELSE
                  prob%DX_u( i ) = zero
                END IF
              END IF
            END DO

!  Parametric constraint bounds

            DO i = 1, prob%m
              IF ( prob%C_l( i ) == prob%C_u( i ) ) THEN
                prob%DC_l( i ) = - one
                prob%DC_u( i ) = - one
              ELSE
                IF ( prob%C_l( i ) >= - infinity ) THEN
                  prob%DC_l( i ) = one
                ELSE
                  prob%DC_l( i ) = zero
                END IF
                IF ( prob%C_u( i ) <= infinity ) THEN
                  prob%DC_u( i ) = - one
                ELSE
                  prob%DC_u( i ) = zero
                END IF
              END IF
            END DO
            prob%DC_u( 1 ) = one
            prob%DX_l( 1 ) = 0.5_wp
            prob%DX_l( 2 ) = zero
!           prob%DC_u( 2 ) = zero
          END IF

!  Parametric bounds

         theta_max = theta_max + 4.0_wp
!        theta_max = theta_max + one
         PQP_control%randomize = .FALSE.
         PQP_control%cold_start = 0
!        PQP_control%each_interval = .FALSE.
        ELSE IF ( pass == n_pass ) THEN
          theta_max = zero
          prob%G = prob%G + theta_u * prob%DG
          prob%X_l = prob%X_l + theta_u * prob%DX_l
          prob%X_u = prob%X_u + theta_u * prob%DX_u
          prob%C_l = prob%C_l + theta_u * prob%DC_l
          prob%C_u = prob%C_u + theta_u * prob%DC_u
        ELSE
          prob%rho_g = rho_g ; prob%rho_b = rho_b
          prob%DG = zero
          prob%DX_l = zero
          prob%DX_u = zero
          prob%DC_l = zero
          prob%DC_u = zero
        END IF
        prob%theta_max = theta_max
!       PQP_control%treat_zero_bounds_as_general = .TRUE.

!       WRITE( out, "( /, ' =================== ', /, '  mu = ', ES12.4,       &
!      &               /, ' =================== ' )" ) mu

        PQP_control%restore_problem = 2
        SCALING_control%print_level = PQP_control%print_level
        SCALING_control%out         = PQP_control%out
        SCALING_control%out_error   = PQP_control%error
!       IF ( pass == n_pass - 1 ) SCALING_control%print_level = 1
!       IF ( pass == n_pass - 1 ) PQP_control%print_level = 1

        printo = out > 0 .AND. SCALING_control%print_level > 0
        printe = out > 0 .AND. SCALING_control%print_level >= 0

        IF ( printo ) CALL COPYRIGHT( out, '2002' )

!  If required, scale the problem

        IF ( scale > 0 ) THEN
          ALLOCATE( SH( n ), SA( m ), STAT = alloc_stat )
          IF ( alloc_stat /= 0 ) THEN
            IF ( printe ) WRITE( out, 2150 ) 'SH/SA', alloc_stat ; STOP
          END IF

!  Scale using K

          CALL SCALING_initialize( SCALING_control )
          IF ( scale == 1 .OR. scale == 4 ) THEN
            IF ( printo ) WRITE( out, 2140 ) 'K'
            CALL SCALING_get_factors_from_K( n, m, prob%H%val, prob%H%col,     &
                                             prob%H%ptr, prob%A%val,           &
                                             prob%A%col, prob%A%ptr, SH, SA,   &
                                             SCALING_control, ifail )

!  Scale using A

          ELSE IF ( scale == 2 .OR. scale == 5 ) THEN
            IF ( printo ) WRITE( out, 2140 ) 'A'
            CALL SCALING_get_factors_from_A( n, m, prob%A%val, prob%A%col,     &
                                             prob%A%ptr, SH, SA,               &
                                             SCALING_control, ifail )
          ELSE IF ( scale == 3 ) THEN
            SH = one ; SA = one
          END IF

!  Reccale A

          IF ( scale >= 3 ) THEN
            IF ( printo ) WRITE( out, 2170 )
            CALL SCALING_normalize_rows_of_A( n, m, prob%A%val, prob%A%col,    &
                                              prob%A%ptr, SH, SA )
          END IF

!  Apply the scaling factors

          CALL SCALING_apply_factors( n, m, prob%H%val, prob%H%col, prob%H%ptr,&
                                      prob%A%val, prob%A%col, prob%A%ptr,      &
                                      prob%G, prob%X, prob%X_l, prob%X_u,      &
                                      prob%C_l, prob%C_u, prob%Y, prob%Z,      &
                                      infinity, SH, SA, .TRUE., DG = prob%DG,  &
                                      DX_l = prob%DX_l, DX_u = prob%DX_u,      &
                                      DC_l = prob%DC_l, DC_u = prob%DC_u )
        END IF

!  If the problem to be output, allocate sufficient space

        IF ( write_initial_sif ) THEN
  
          ALLOCATE( prob%X_status( n ), STAT = alloc_stat )
          IF ( alloc_stat /= 0 ) THEN
            IF ( printe ) WRITE( out, 2150 ) 'X_status', alloc_stat ; STOP
          END IF
          prob%X_status = ACTIVE
          
          ALLOCATE( prob%C_status( m ), STAT = alloc_stat )
          IF ( alloc_stat /= 0 ) THEN
            IF ( printe ) WRITE( out, 2150 ) 'X_status', alloc_stat ; STOP
          END IF
          prob%C_status =  ACTIVE
          
          ALLOCATE( prob%Z_l( n ), prob%Z_u( n ), STAT = alloc_stat )
          IF ( alloc_stat /= 0 ) THEN
            IF ( printe ) WRITE( out, 2150 ) 'Z_lu', alloc_stat ; STOP
          END IF
          prob%Z_l( : n ) = - infinity
          prob%Z_u( : n ) =   infinity
          
          ALLOCATE( prob%Y_l( m ), prob%Y_u( m ), STAT = alloc_stat )
          IF ( alloc_stat /= 0 ) THEN
            IF ( printe ) WRITE( out, 2150 ) 'C_lu', alloc_stat ; STOP
          END IF
          prob%Y_l( : m ) = - infinity
          prob%Y_u( : m ) =   infinity

!  Writes the initial SIF file, if needed

          IF ( write_initial_sif ) THEN
            CALL QPT_write_to_sif( prob, pname, ifilename, ifiledevice,        &
                                   .FALSE., .FALSE., infinity )
            IF ( .NOT. do_solve ) STOP
          END IF
        END IF

!  ==================
!  Call the optimizer
!  ==================

        CALL CPU_TIME( timeo )
  
        IF ( do_solve .AND. prob%n > 0 ) THEN

          prob%m = m
          prob%n = n

          solv = ' PQP'
          IF ( pass > 1 ) THEN
            B_stat_old( : n ) =  B_stat( : n ) 
            C_stat_old( : m ) =  C_stat( : m )
          END IF

!         IF ( pass > 1 ) PQP_control%print_level = 1
!         PQP_control%print_level = 1

          action = "start"
          WRITE( out, " ( /, ' ** PQP solver used ** ' ) " )
          CALL PQP_solve( action, prob, PQP_interval, C_stat, B_stat, &
                               PQP_data, PQP_control, PQP_inform )
          WRITE( out, " ( ' ** PQP solver exit ** ', / ) " )


          qfval = PQP_inform%obj
!         nmods = 0 ; newton = 0
          theta_u = zero
          IF ( pass == 1 ) X0 = prob%X
          IF ( pass == 2 ) theta_u = PQP_interval%theta_u
          IF ( pass == 2 ) theta_end = PQP_interval%theta_u
!         IF ( pass == 2 ) WRITE( 6, * ) PQP_interval%DX
!         IF ( pass == 3 ) WRITE( 6, * ) ( prob%X - X0 ) / theta_end

!  ====================
!  Retun from optimizer
!  ====================

          CALL CPU_TIME( timet )

!  Deallocate arrays from the minimization

          status = PQP_inform%status ; iter = PQP_inform%iter
          nfacts = PQP_inform%nfacts
          factorization_integer = PQP_inform%factorization_integer 
          factorization_real = PQP_inform%factorization_real
          CALL PQP_terminate( PQP_data, PQP_control,            &
                                   PQP_inform )
        ELSE
          timeo  = 0.0
          timet  = 0.0
          iter   = 0
          solv   = ' NONE'
          status = 0
          nfacts = 0
          factorization_integer = 0
          factorization_real    = 0.0
!         newton = 0
!         nmods  = 0
!         qfval  = prob%f
        END IF

!  If the problem was scaled, unscale it.

        IF ( scale > 0 ) THEN
          CALL SCALING_apply_factors( n, m, prob%H%val, prob%H%col, prob%H%ptr,&
                                      prob%A%val, prob%A%col, prob%A%ptr,      &
                                      prob%G, prob%X, prob%X_l, prob%X_u,      &
                                      prob%C_l, prob%C_u, prob%Y, prob%Z,      &
                                      infinity, SH, SA, .FALSE., DG = prob%DG, &
                                      DX_l = prob%DX_l, DX_u = prob%DX_u,      &
                                      DC_l = prob%DC_l, DC_u = prob%DC_u )
          DEALLOCATE( SH, SA )
        END IF

!  If required, compute the exit status for the variables and constraints

        IF ( pass > 1 ) THEN
          miss_ident = COUNT( B_stat_old( : n ) /=  B_stat( : n ) ) +          &
                       COUNT( C_stat_old( : m ) /=  C_stat( : m ) )
          WRITE( 6, "( ' # misidentified statii = ', I7 )" ) miss_ident
        END IF

!  Print details of the solution obtained

        timet = timet - timeo
        IF ( print_solution ) THEN
          WRITE( out, 2070 ) pname 
          WRITE( out, 2010 ) status
          WRITE( out, 2060 ) times + timet 
          WRITE( out, 2030 ) qfval, iter, nfacts, factorization_integer,       &
                                                  factorization_real 
        END IF
        IF ( status == 0 .OR. status == - 8 .OR. status == - 9 .OR.            &
               status == - 10 ) THEN
          IF ( print_solution ) THEN

            l = 4 ; IF ( fulsol ) l = n 

!  Print details of the primal and dual variables

            WRITE( out, 2090 ) 
            DO j = 1, 2 
              IF ( j == 1 ) THEN 
                ir = 1 ; ic = MIN( l, n ) 
              ELSE 
                IF ( ic < n - l ) WRITE( out, 2000 ) 
                ir = MAX( ic + 1, n - ic + 1 ) ; ic = n 
              END IF 

              DO i = ir, ic 
                state = ' FREE' 
                xl = prob%X_l( i ) + theta_u * prob%DX_l( i )
                xu = prob%X_u( i ) + theta_u * prob%DX_u( i )
                IF ( ABS( prob%X  ( i ) - xl ) < ten * stopr )                 &
                  state = 'LOWER'
                IF ( ABS( prob%X  ( i ) - xu ) < ten * stopr )                 &
                  state = 'UPPER'
                IF ( ABS( xl - xu ) < stopr )                                  &
                  state = 'FIXED'
                WRITE( out, 2050 ) i, VNAME( i ), state, prob%X( i ),          &
                                   xl, xu, prob%Z( i )
              END DO 
            END DO 
          END IF

!  Compute the number of fixed and degenerate variables.

          nfixed = 0 ; ndegen = 0 
          DO i = 1, n 
            xl = prob%X_l( i ) + theta_u * prob%DX_l( i )
            xu = prob%X_u( i ) + theta_u * prob%DX_u( i )
            IF ( ABS( prob%X( i ) - xl ) < ten * stopr ) THEN
              nfixed = nfixed + 1 
              IF ( ABS( prob%Z( i ) ) < ten * stopr ) ndegen = ndegen + 1 
            ELSE IF ( ABS( prob%X( i ) - xu ) < ten * stopr ) THEN
              nfixed = nfixed + 1 
              IF ( ABS( prob%Z( i ) ) < ten * stopr ) ndegen = ndegen + 1 
            END IF 
          END DO 

!  Print details of the constraints.

          IF ( print_solution ) THEN
            IF ( m > 0 ) THEN 
    
              WRITE( out, 2040 ) 
              l = 2  ; IF ( fulsol ) l = m 
              DO j = 1, 2 
                IF ( j == 1 ) THEN 
                  ir = 1 ; ic = MIN( l, m ) 
                ELSE 
                  IF ( ic < m - l ) WRITE( out, 2000 ) 
                  ir = MAX( ic + 1, m - ic + 1 ) ; ic = m 
                END IF 
                DO i = ir, ic 
                  state = ' FREE' 
                  cl = prob%C_l( i ) + theta_u * prob%DC_l( i )
                  cu = prob%C_u( i ) + theta_u * prob%DC_u( i )
                  IF ( ABS( prob%C( I ) - cl ) < ten * stopr )                 &
                    state = 'LOWER' 
                  IF ( ABS( prob%C( I ) - cu ) < ten * stopr )                 &
                    state = 'UPPER' 
                  IF ( ABS( cl - cu ) < stopr )                                &
                    state = 'EQUAL' 
                  WRITE( out, 2130 ) i, CNAME( i ), STATE, prob%C( i ),        &
                                     cl, cu, prob%Y( i ) 
                END DO 
              END DO 
            END IF 
          END IF

!  Compute the number of equality, fixed inequality and degenerate constraints

          mequal = 0 ; mfixed = 0 ; mdegen = 0 
          DO i = 1, m 
            cl = prob%C_l( i ) + theta_u * prob%DC_l( i )
            cu = prob%C_u( i ) + theta_u * prob%DC_u( i )
            IF ( ABS( prob%C( i ) - cl ) < ten * stopr .OR.                    &
                 ABS( prob%C( i ) - cu ) < ten * stopr ) THEN
              IF ( ABS( cl - cu ) < ten * stopr ) THEN 
                 mequal = mequal + 1 
              ELSE 
                 mfixed = mfixed + 1 
              END IF 
              IF ( ABS( prob%Y( i ) ) < stopr ) mdegen = mdegen + 1 
            END IF 
          END DO 
          WRITE( out, 2100 ) n, nfixed, ndegen 
          IF ( m > 0 ) THEN 
             WRITE( out, 2110 ) m, mequal, mdegen 
             IF ( m /= mequal ) WRITE( out, 2120 ) mfixed 
          END IF 

!  If required, write the solution to a file

          IF ( write_solution ) THEN
            INQUIRE( FILE = sfilename, EXIST = filexx )
            IF ( filexx ) THEN
               OPEN( sfiledevice, FILE = sfilename, FORM = 'FORMATTED',        &
                   STATUS = 'OLD', IOSTAT = iores )
            ELSE
               OPEN( sfiledevice, FILE = sfilename, FORM = 'FORMATTED',        &
                    STATUS = 'NEW', IOSTAT = iores )
            END IF
            IF ( iores /= 0 ) THEN 
              write( out, 2160 ) iores, sfilename
              STOP
            END IF
  
            WRITE( sfiledevice, 2250 ) pname, solv, qfval
            WRITE( sfiledevice, 2090 ) 
  
            DO i = 1, n 
              state = ' FREE' 
              xl = prob%X_l( i ) + theta_u * prob%DX_l( i )
              xu = prob%X_u( i ) + theta_u * prob%DX_u( i )
              IF ( ABS( prob%X( i )   - xl ) < ten * stopr )                   &
                state = 'LOWER' 
              IF ( ABS( prob%X( i )   - xu ) < ten * stopr )                   &
                state = 'UPPER' 
              IF ( ABS( xl - xu ) < stopr )                                    &
                state = 'FIXED' 
              WRITE( sfiledevice, 2050 ) i, VNAME( i ), STATE, prob%X( i ),    &
                xl, xu, prob%Z( i )
            END DO 
    
            IF ( m > 0 ) THEN 
              WRITE( sfiledevice, 2040 ) 
              DO i = 1, m 
                state = ' FREE' 
                cl = prob%C_l( i ) + theta_u * prob%DC_l( i )
                cu = prob%C_u( i ) + theta_u * prob%DC_u( i )
                IF ( ABS( prob%C( I ) - cl ) < ten * stopr )                   &
                  state = 'LOWER'
                IF ( ABS( prob%C( I ) - cu ) < ten * stopr )                   &
                  state = 'UPPER'
                IF ( ABS( cl - cu ) < stopr )                                  &
                  state = 'EQUAL' 
                WRITE( sfiledevice, 2130 ) i, CNAME( i ), STATE, prob%C( i ),  &
                  cl, cu, prob%C_u( i ), prob%Y( i )   
              END DO 
            END IF 
    
            WRITE( sfiledevice, 2030 ) qfval, iter, nfacts,                    &
                   factorization_integer, factorization_real 
            CLOSE( sfiledevice ) 
          END IF 
        END IF 

!  Compare the variants used so far

        WRITE( out, 2190 )
        WRITE( out, 2080 ) solv, iter, nfacts, qfval, status, times, timet,    &
                           times + timet 

        s_iter( pass ) = iter
        s_status( pass ) = status
        s_qfval( pass ) = qfval
        s_timet( pass ) = timet
        IF ( write_result_summary ) THEN
          IF ( pass > 1 ) THEN
            BACKSPACE( rfiledevice )
            WRITE( rfiledevice, 2300 ) pname, n, m, miss_ident,                &
              s_iter( pass ), s_qfval( pass ), s_status( pass ), s_timet( pass )
          END IF
        END IF

!  ================
!  End of Main loop
!  ================

      END DO
      IF ( is_specfile ) CLOSE( input_specfile )
      DEALLOCATE( VNAME, CNAME, C )
      CALL CUTEST_cterminate( cutest_status )
      RETURN

  910 CONTINUE
      WRITE( out, "( ' CUTEst error, status = ', i0, ', stopping' )" )         &
        cutest_status
      status = - 98
      RETURN

!  Non-executable statements

 2000 FORMAT( '      . .          .....  ..........',                          &
              '  ..........  ..........  .......... ' ) 
 2010 FORMAT( /,' Stopping with inform%status = ', I3, / ) 
 2020 FORMAT( /, ' Problem: ', A10 )
 2030 FORMAT( /,' Final objective function value ', ES22.14, /,                &
          ' Total number of iterations = ',I6,' Number of factorizations = ',  &
          I6, //, I10, ' integer and ', I10, ' real words required',           &
          ' for the factorization' ) 
 2040 FORMAT( /,' Constraints : ', /, '                              ',        &
                '        <------ Bounds ------> ', /                           &
                '      # name       state    value   ',                        &
                '    Lower       Upper     Multiplier ' ) 
 2050 FORMAT( I7, 1X, A10, A6, 4ES12.4 ) 
 2060 FORMAT( /, ' Total time = ', 0P, F12.2 ) 
 2070 FORMAT( /, ' Problem: ', A10 )
 2190 FORMAT( /,  '                                     objective',            &
                  '          < ------ time ----- > ', /,                       &
                  ' Method      iterations   factors      value  ',            &
                  '   status setup PQP  total', /,                             &
                  ' ------      ----------   -------    ---------',            &
                  '   ------ -----    ----   -----  ' ) 
 2080 FORMAT( A9, 2I10, 6X, ES12.4, I6, 0P, 3F8.2 ) 
 2090 FORMAT( /,' Solution : ', /,'                              ',            &
                '        <------ Bounds ------> ', /                           &
                '      # name       state    value   ',                        &
                '    Lower       Upper       Dual ' ) 
 2100 FORMAT( /, ' Of the ', I7, ' variables ', 2X, I7,                        &
              ' are on bounds &', I7, ' are dual degenerate' ) 
 2110 FORMAT( ' Of the ', I7, ' constraints ', I7,' are equations &', I7,      &
              ' are degenerate' ) 
 2120 FORMAT( ' Of the inequality constraints ', I6, ' are on bounds' ) 
 2130 FORMAT( I7, 1X, A10, A6, 4ES12.4 ) 
 2140 FORMAT( /, ' *** Problem will be scaled based on ', A1, ' *** ' )
 2150 FORMAT( ' Allocation error, variable ', A8, ' status = ', I6 )
 2160 FORMAT( ' IOSTAT = ', I6, ' when opening file ', A9, '. Stopping ' )
 2170 FORMAT( /, ' *** Further scaling applied to A *** ' )
 2180 FORMAT( A10 )
!2190 FORMAT( A10, A5, 2I7, 3I6, ES13.4, I6, 0P, F8.2 ) 
 2250 FORMAT( /, ' Problem:    ', A10, /, ' Solver :   ', A5,                  &
              /, ' Objective:', ES24.16 )
 2300 FORMAT( A10, 2I7, I6,    I6, ES13.4, I6, 0P, F8.1   )

!  End of subroutine USE_PARAM

      END SUBROUTINE USE_PQP

!  End of module USEPQP_double

    END MODULE GALAHAD_USEPQP_double
