! THIS VERSION: GALAHAD 2.5 - 09/02/2013 AT 16:05 GMT.

!-*-*-*-*-*-*-*-*-*-  G A L A H A D   R U N L P Q P B   *-*-*-*-*-*-*-*-*-*-*-

!  Nick Gould, for GALAHAD productions
!  Copyright reserved
!  July 29th 2002

      PROGRAM RUNLPQPB

!  Main program for LPQPB, an interior point algorithm for solving l_p 
!  quadratic programs

      USE CUTEst_interface_double
!NOT95USE GALAHAD_CPU_time
      USE GALAHAD_QPT_double
      USE GALAHAD_LPQPB_double
      USE GALAHAD_SORT_double, only: SORT_reorder_by_rows
      USE GALAHAD_PRESOLVE_double
      USE GALAHAD_SPECFILE_double 
      USE SCALING
      USE GALAHAD_SYMBOLS,                                                     &
          ACTIVE                => GALAHAD_ACTIVE,                             &
          DEBUG                 => GALAHAD_DEBUG,                              &
          GENERAL               => GALAHAD_GENERAL,                            &
          ALL_ZEROS             => GALAHAD_ALL_ZEROS

      IMPLICIT NONE

!  --------------------------------------------------------------------
!
!      Solve the l_1 quadratic program               
!                                                    
!        minimize     1/2 x(T) H x + g(T) x + f
!                       + rho_g min( A x - c_l , 0 )
!                       + rho_g max( A x - c_u , 0 )
!                       + rho_b min( x - x_l , 0 )
!                       + rho_b max( x - x_u , 0 )
!
!      using the GALAHAD package GALAHAD_QPB
!
!  --------------------------------------------------------------------

!  Parameters

      INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
      INTEGER, PARAMETER :: long = SELECTED_INT_KIND( 18 )
      REAL ( KIND = wp ), PARAMETER :: zero = 0.0_wp
      REAL ( KIND = wp ), PARAMETER :: one = 1.0_wp
      REAL ( KIND = wp ), PARAMETER :: ten = 10.0_wp
      REAL ( KIND = wp ), PARAMETER :: infinity = ten ** 19
      REAL ( KIND = wp ), PARAMETER :: biginf = HUGE( one )

!     INTEGER, PARAMETER :: n_k = 100, k_k = 3, in = 28
!     REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( :, : ) :: k_val
!     CHARACTER ( len = 10 ) :: filename = 'k.val'

!  Scalars

      INTEGER :: i, j, l, neh, nea, n, m, ir, ic, la, lh, liw, iores
      INTEGER :: status, mfixed, mdegen, iter, nfacts, nfixed, ndegen, mequal
      INTEGER :: alloc_stat, cutest_status, newton, nmods, A_ne, H_ne
      INTEGER ( KIND = long ) :: factorization_integer, factorization_real
      REAL :: time, timeo, times, timet
      REAL ( KIND = wp ) :: obj, qfval, stopr, dummy, res_c, res_k, max_cs
      LOGICAL :: filexx, printo, printe
      
!  Problem input characteristics

      INTEGER, PARAMETER :: input = 55
      CHARACTER ( LEN = 16 ) :: prbdat = 'OUTSDIF.d'

!  Specfile characteristics

      INTEGER, PARAMETER :: input_specfile = 34
      INTEGER, PARAMETER :: lspec = 21
      CHARACTER ( LEN = 16 ) :: specname = 'RUNLPQPB'
      TYPE ( SPECFILE_item_type ), DIMENSION( lspec ) :: spec
      CHARACTER ( LEN = 16 ) :: runspec = 'RUNLPQPB.SPC'

!  Default values for specfile-defined parameters

      INTEGER :: scale = 0
      INTEGER :: dfiledevice = 26
      INTEGER :: ifiledevice = 51
      INTEGER :: pfiledevice = 53
      INTEGER :: rfiledevice = 47
      INTEGER :: sfiledevice = 62
!     LOGICAL :: write_problem_data   = .FALSE.
      LOGICAL :: write_problem_data   = .TRUE.
      LOGICAL :: write_initial_sif    = .FALSE.
      LOGICAL :: write_presolved_sif  = .FALSE.
      LOGICAL :: write_solution       = .FALSE.
      LOGICAL :: write_result_summary = .FALSE.
      CHARACTER ( LEN = 30 ) :: dfilename = 'LPQPB.data'
      CHARACTER ( LEN = 30 ) :: ifilename = 'ORIG.SIF'
      CHARACTER ( LEN = 30 ) :: pfilename = 'PRE.SIF'
      CHARACTER ( LEN = 30 ) :: rfilename = 'LPQPBRES.d'
      CHARACTER ( LEN = 30 ) :: sfilename = 'LPQPBSOL.d'
      LOGICAL :: do_presolve = .TRUE.
      LOGICAL :: do_solve = .TRUE.
      LOGICAL :: fulsol = .FALSE. 
      REAL ( KIND = wp ) :: rho = 100000.0
      LOGICAL :: one_norm = .FALSE.

!  Output file characteristics

      INTEGER, PARAMETER :: out  = 6
      INTEGER, PARAMETER :: io_buffer = 11
      INTEGER :: errout = 6
      CHARACTER ( LEN =  5 ) :: state, solv = 'LPQPB'
      CHARACTER ( LEN = 10 ) :: pname

!  Arrays

      TYPE ( QPT_problem_type ) :: prob
      TYPE ( LPQPB_control_type ) :: control        
      TYPE ( LPQPB_inform_type ) :: inform
      TYPE ( LPQPB_data_type ) :: data

!  Allocatable arrays

      CHARACTER ( LEN = 10 ), ALLOCATABLE, DIMENSION( : ) :: VNAME, CNAME
      REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: X0, C
      REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: AY, HX
      LOGICAL, ALLOCATABLE, DIMENSION( : ) :: EQUATN, LINEAR
      INTEGER, ALLOCATABLE, DIMENSION( : ) :: IW

!  Open the relevant file.

      OPEN( input, FILE = prbdat, FORM = 'FORMATTED', STATUS = 'OLD'  )
      REWIND input

      CALL CPU_TIME( time )
      printe = out > 0 .AND. control%print_level >= 0

!  Determine the number of variables and constraints

      CALL CUTEST_cdimen( cutest_status, input, n, m )
      IF ( cutest_status /= 0 ) GO TO 910

!  Allocate suitable arrays

      ALLOCATE( prob%X( n ), prob%X_l( n ), prob%X_u( n ),                     &
                prob%G( n ), VNAME( n ), STAT = alloc_stat )
      IF ( alloc_stat /= 0 ) THEN
        WRITE( out, 2150 ) 'X', alloc_stat ; STOP
      END IF

      ALLOCATE( prob%C_l( m ), prob%C_u( m ), prob%Y( m ), CNAME( m ),         &
                EQUATN( m ), LINEAR( m ), STAT = alloc_stat )
      IF ( alloc_stat /= 0 ) THEN
        WRITE( out, 2150 ) 'C', alloc_stat ; STOP
      END IF

!  Set up the data structures necessary to hold the group partially
!  separable function.

      CALL CUTEST_csetup( cutest_status, input, out, io_buffer,                &
                          n, m, prob%X, prob%X_l, prob%X_u,                    &
                          prob%Y, prob%C_l, prob%C_u, EQUATN, LINEAR,  0, 0, 0 )
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

!  Allocate and initialize dual variables.

      ALLOCATE( prob%Z( n ), STAT = alloc_stat )
      IF ( alloc_stat /= 0 ) THEN
        WRITE( out, 2150 ) 'Z', alloc_stat
        STOP
      END IF
      prob%Z( : n ) = one

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

!  Evaluate the constant terms of the objective (obj) and constraint 
!  functions (C)

      CALL CUTEST_cfn( cutest_status, n, m, X0, obj, C( : m ) )
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

!  Determine the number of nonzeros in the Jacobian

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

      CALL CUTEST_csgr( cutest_status, n, m, X0, prob%Y, .FALSE.,              &
                        nea, la, prob%A%val, prob%A%col, prob%A%row )
      IF ( cutest_status /= 0 ) GO TO 910
      DEALLOCATE( X0 )
      
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
      lh = MAX( lh, 1 )

!  Allocate arrays to hold the Hessian

      ALLOCATE( prob%H%row( lh ), prob%H%col( lh ), prob%H%val( lh ),          &
                STAT = alloc_stat )
      IF ( alloc_stat /= 0 ) THEN
        WRITE( out, "( ' nea = ', i8, ' la   = ', i8 )" ) nea, la
        WRITE( out, 2150 ) 'H', alloc_stat
        STOP
      END IF

!  Evaluate the Hessian of the Lagrangian function at the initial point.

      CALL CUTEST_csh( cutest_status, n, m, prob%X, prob%Y,                    &
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

      liw = MAX( m, n ) + 1
      ALLOCATE( prob%A%ptr( m + 1 ), prob%H%ptr( n + 1 ) )
      ALLOCATE( IW( liw ) )

!  Transform A to row storage format

      IF ( A_ne /= 0 ) THEN
        CALL SORT_reorder_by_rows( m, n, A_ne, prob%A%row, prob%A%col, A_ne,   &
                                   prob%A%val, prob%A%ptr, m + 1, IW, liw,     &
                                   out, out, i )
      ELSE
        prob%A%ptr = 1
      END IF

!  Same for H

      IF ( H_ne /= 0 ) THEN
        CALL SORT_reorder_by_rows( n, n, H_ne, prob%H%row, prob%H%col, H_ne,   &
                                   prob%H%val, prob%H%ptr, n + 1, IW, liw,     &
                                   out, out, i )
      ELSE
        prob%H%ptr = 1
      END IF

!  Deallocate arrays holding matrix row indices

      DEALLOCATE( prob%A%row, prob%H%row )
      DEALLOCATE( IW )
      ALLOCATE( prob%A%row( 0 ), prob%H%row( 0 ) )

!  Store the problem dimensions

      prob%n = n
      prob%m = m
      prob%A%ne = - 1
      prob%H%ne = - 1
      prob%f = obj ; prob%rho_g = 2 * m ; prob%rho_b = 2 * n
      CLOSE( input  )

!  ------------------- problem set-up complete ----------------------

      CALL CPU_TIME( times )

!  ------------------ Open the specfile for runqpb ----------------

      OPEN( input_specfile, FILE = runspec, FORM = 'FORMATTED', STATUS = 'OLD' )

!   Define the keywords

      spec( 1 )%keyword = 'write-problem-data'
      spec( 2 )%keyword = 'problem-data-file-name'
      spec( 3 )%keyword = 'problem-data-file-device'
      spec( 4 )%keyword = 'write-initial-sif'
      spec( 5 )%keyword = 'initial-sif-file-name'
      spec( 6 )%keyword = 'initial-sif-file-device'
      spec( 7 )%keyword = 'rho-used'
      spec( 8 )%keyword = 'scale-problem'
      spec( 9 )%keyword = 'pre-solve-problem'
      spec( 10 )%keyword = 'write-presolved-sif'
      spec( 11 )%keyword = 'presolved-sif-file-name'
      spec( 12 )%keyword = 'presolved-sif-file-device'
      spec( 13 )%keyword = 'solve-problem'
      spec( 14 )%keyword = 'print-full-solution'
      spec( 15 )%keyword = 'write-solution'
      spec( 16 )%keyword = 'solution-file-name'
      spec( 17 )%keyword = 'solution-file-device'
      spec( 18 )%keyword = 'write-result-summary'
      spec( 19 )%keyword = 'result-summary-file-name'
      spec( 20 )%keyword = 'result-summary-file-device'
      spec( 21 )%keyword = 'one-norm-penalty'

!   Read the specfile

      CALL SPECFILE_read( input_specfile, specname, spec, lspec, errout )

!   Interpret the result

      CALL SPECFILE_assign_logical( spec( 1 ), write_problem_data, errout )
      CALL SPECFILE_assign_string ( spec( 2 ), dfilename, errout )
      CALL SPECFILE_assign_integer( spec( 3 ), dfiledevice, errout )
      CALL SPECFILE_assign_logical( spec( 4 ), write_initial_sif, errout )
      CALL SPECFILE_assign_string ( spec( 5 ), ifilename, errout )
      CALL SPECFILE_assign_integer( spec( 6 ), ifiledevice, errout )
      CALL SPECFILE_assign_real( spec( 7 ), rho, errout )
      CALL SPECFILE_assign_integer( spec( 8 ), scale, errout )
      CALL SPECFILE_assign_logical( spec( 9 ), do_presolve, errout )
      CALL SPECFILE_assign_logical( spec( 10 ), write_presolved_sif, errout )
      CALL SPECFILE_assign_string ( spec( 11 ), pfilename, errout )
      CALL SPECFILE_assign_integer( spec( 12 ), pfiledevice, errout )
      CALL SPECFILE_assign_logical( spec( 13 ), do_solve, errout )
      CALL SPECFILE_assign_logical( spec( 14 ), fulsol, errout )
      CALL SPECFILE_assign_logical( spec( 15 ), write_solution, errout )
      CALL SPECFILE_assign_string ( spec( 16 ), sfilename, errout )
      CALL SPECFILE_assign_integer( spec( 17 ), sfiledevice, errout )
      CALL SPECFILE_assign_logical( spec( 18 ), write_result_summary, errout )
      CALL SPECFILE_assign_string ( spec( 19 ), rfilename, errout )
      CALL SPECFILE_assign_integer( spec( 20 ), rfiledevice, errout )
      CALL SPECFILE_assign_logical( spec( 21 ), one_norm, errout )

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
        WRITE( dfiledevice, "( ' A_ptr ', /, ( 10I6 ) )" )                     &
          prob%A%ptr( : m + 1 )
        WRITE( dfiledevice, "( ' A_col ', /, ( 10I6 ) )" )                     &
          prob%A%col( : A_ne )
        WRITE( dfiledevice, "( ' A_val ', /, ( 5ES12.4 ) )" )                  &
          prob%A%val( : A_ne )
        WRITE( dfiledevice, "( ' H_ptr ', /, ( 10I6 ) )" )                     &
          prob%H%ptr( : n + 1 )
        WRITE( dfiledevice, "( ' H_col ', /, ( 10I6 ) )" )                     &
          prob%H%col( : H_ne )
        WRITE( dfiledevice, "( ' H_val ', /, ( 5ES12.4 ) )" )                  &
          prob%H%val( : H_ne )

        CLOSE( dfiledevice )
      END IF

!  Writes the initial SIF file, if needed

      IF ( write_initial_sif ) THEN
        ALLOCATE( prob%X_status( n ), STAT = alloc_stat )
        IF ( alloc_stat /= 0 ) THEN
          IF ( printe ) WRITE( out, 2150 ) 'X_status', alloc_stat
          STOP
        END IF
        prob%X_status = ACTIVE
        
        ALLOCATE( prob%C_status( m ), STAT = alloc_stat )
        IF ( alloc_stat /= 0 ) THEN
          IF ( printe ) WRITE( out, 2150 ) 'C_status', alloc_stat ; STOP
        END IF
        prob%C_status = ACTIVE

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

        CALL QPT_write_to_sif( prob, pname, ifilename, ifiledevice,            &
                               .FALSE., .FALSE., infinity )
      END IF

!  Update control parameters if required.

      CALL LPQPB_initialize( data, control, inform )
      CALL LPQPB_read_specfile( control, input_specfile )

      printo = out > 0 .AND. control%QPB_control%print_level > 0
      printe = out > 0 .AND. control%QPB_control%print_level >= 0

      IF ( printo ) WRITE( out, "(                                             &
     &         /, ' Copyright GALAHAD productions, 2002',                      &
     &         //, ' - Use of this code is restricted to those who',           &
     &         /,  ' - agree to abide by the conditions-of-use set out',       &
     &         /,  ' - in the README.cou file distributed with the',           &
     &         /,  ' - source to the GALAHAD codes or from the WWW at',        &
     &         /,  ' - http://galahad.rl.ac.uk/galahad-www/cou.html', / )" )

!  Solve the problem
!  =================


      control%QPB_control%LSQP_control%pivot_tol = control%QPB_control%pivot_tol
      control%QPB_control%LSQP_control%pivot_tol_for_dependencies =            &
        control%QPB_control%pivot_tol_for_dependencies
!     control%QPB_control%LSQP_control%maxit = 1
!     control%QPB_control%LSQP_control%print_level = 3
      control%QPB_control%restore_problem = 2

!  Call the optimizer

      qfval = obj 

      IF ( do_solve .AND. prob%n > 0 ) THEN

        CALL CPU_TIME( timeo )
  
!       prob%m = m
!       prob%n = n
  
!       WRITE( 6, "( ' x ', /, (5ES12.4) )" ) prob%X
!       WRITE( 6, "( ' y ', /, (5ES12.4) )" ) prob%Y
!       WRITE( 6, "( ' z ', /, (5ES12.4) )" ) prob%Z
  
        solv = 'LPQPB'
        IF ( printo ) WRITE( out, " ( /, ' ** QPB solver used ** ' ) " )
        CALL LPQPB_solve( prob, rho, one_norm, data, control, inform )

        IF ( printo ) WRITE( out, " ( /, ' ** LPQPB solver used ** ' ) " )
        qfval = inform%QPB_inform%obj 
        nmods = inform%QPB_inform%nmods ; newton = 0
  
        CALL CPU_TIME( timet )
  
!  Deallocate arrays from the minimization
  
        status = inform%QPB_inform%status ; iter = inform%QPB_inform%iter
        nfacts = inform%QPB_inform%nfacts ; stopr = control%QPB_control%stop_d
        factorization_integer = inform%QPB_inform%factorization_integer 
        factorization_real = inform%QPB_inform%factorization_real
      ELSE
        timeo  = 0.0
        timet  = 0.0
        iter   = 0
        solv   = ' NONE'
        status = 0
        stopr  = control%QPB_control%stop_d
        nfacts = 0
        factorization_integer = 0
        factorization_real    = 0
        newton = 0
        nmods  = 0
        qfval  = prob%f
      END IF

!  Compute maximum contraint residual and complementary slackness

      res_c = zero ; max_cs = zero
      DO i = 1, prob%m
        dummy = zero
        DO j = prob%A%ptr( i ), prob%A%ptr( i + 1 ) - 1
          dummy = dummy + prob%A%val( j ) * prob%X( prob%A%col( j ) )
        END DO
        res_c = MAX( res_c, MAX( zero, prob%C_l( i ) - dummy,                  &
                                       dummy - prob%C_u( i ) ) )
        IF ( prob%C_l( i ) > - infinity ) THEN
          IF ( prob%C_u( i ) < infinity ) THEN
            max_cs = MAX( max_cs,                                              &
              MIN( ABS( ( prob%C_l( i ) - dummy ) * prob%Y( i ) ),             &
                   ABS( ( prob%C_u( i ) - dummy ) * prob%Y( i ) ) ) ) 
          ELSE
            max_cs = MAX( max_cs,                                              &
                   ABS( ( prob%C_l( i ) - dummy ) * prob%Y( i ) ) )
          END IF
        ELSE IF ( prob%C_u( i ) < infinity ) THEN
          max_cs = MAX( max_cs,                                                &
                    ABS( ( prob%C_u( i ) - dummy ) * prob%Y( i ) ) )
        END IF
      END DO

      DO i = 1, prob%n
        dummy = prob%X( i )
        IF ( prob%X_l( i ) > - infinity ) THEN
          IF ( prob%X_u( i ) < infinity ) THEN
            max_cs = MAX( max_cs,                                              &
              MIN( ABS( ( prob%X_l( i ) - dummy ) * prob%Z( i ) ),             &
                   ABS( ( prob%X_u( i ) - dummy ) * prob%Z( i ) ) ) )
          ELSE
            max_cs = MAX( max_cs,                                              &
                      ABS( ( prob%X_l( i ) - dummy ) * prob%Z( i ) ) )
          END IF
        ELSE IF ( prob%X_u( i ) < infinity ) THEN
          max_cs = MAX( max_cs,                                                &
                      ABS( ( prob%X_u( i ) - dummy ) * prob%Z( i ) ) )
        END IF
      END DO

!  Compute maximum KKT residual

      ALLOCATE( AY( prob%n ), HX( prob%n ), STAT = alloc_stat )
      AY = zero ; HX = prob%G( : prob%n )
!     prob%G( : prob%n ) 
!       = prob%G( : prob%n ) - prob%Z( : prob%n )
      DO i = 1, prob%m
        DO l = prob%A%ptr( i ), prob%A%ptr( i + 1 ) - 1
          j = prob%A%col( l )
!         prob%G( j ) =                                                        &
!           prob%G( j ) - prob%A%val( l ) * prob%Y( i )
          AY( j ) = AY( j ) - prob%A%val( l ) * prob%Y( i )
        END DO
      END DO
      DO i = 1, prob%n
        DO l = prob%H%ptr( i ), prob%H%ptr( i + 1 ) - 1
          j = prob%H%col( l )
!         prob%G( i ) =                                                        &
!           prob%G( i ) + prob%H%val( l ) * prob%X( j )
!         IF ( j /= i ) prob%G( j ) =                                          &
!            prob%G( j ) + prob%H%val( l ) * prob%X( i )
          HX( i ) = HX( i ) + prob%H%val( l ) * prob%X( j )
          IF ( j /= i )                                                        &
            HX( j ) = HX( j ) + prob%H%val( l ) * prob%X( i )
        END DO
      END DO
!     DO i = 1, n
!       WRITE(6,"( i6, 4ES12.4 )" ) i, HX( i ), prob%Z( i ), AY( i ),          &
!                                   HX( i ) - prob%Z( i ) + AY( i )
!     END DO
!     WRITE(6,"( ( 5ES12.4 ) ) " ) MAXVAL( ABS( prob%Z ) )
!     WRITE(6,"( ' G ', /, ( 5ES12.4 ) )" ) prob%G( : n )
      res_k = MAXVAL( ABS( HX( : n ) - prob%Z( : n ) + AY( : n ) ) ) 

!  Print details of the solution obtained

      WRITE( out, 2010 ) status
      IF ( status == 0 .OR. status == - 8 .OR. status == - 9 .OR.              &
           status == - 10 ) THEN
        l = 4
        IF ( fulsol ) l = prob%n 
        IF ( do_presolve ) THEN
          IF ( control%QPB_control%print_level >= DEBUG ) l = prob%n
        END IF

!  Print details of the primal and dual variables

        WRITE( out, 2090 ) 
        DO j = 1, 2 
          IF ( j == 1 ) THEN 
            ir = 1 ; ic = MIN( l, prob%n ) 
          ELSE 
            IF ( ic < prob%n - l ) WRITE( out, 2000 ) 
            ir = MAX( ic + 1, prob%n - ic + 1 ) ; ic = prob%n 
          END IF 
          DO i = ir, ic 
            state = ' FREE' 
            IF ( ABS( prob%X  ( i ) - prob%X_l( i ) ) < ten * stopr )          &
              state = 'LOWER'
            IF ( ABS( prob%X  ( i ) - prob%X_u( i ) ) < ten * stopr )          &
              state = 'UPPER'
            IF ( ABS( prob%X_l( i ) - prob%X_u( i ) ) <     1.0D-10 )          &
              state = 'FIXED'
            IF ( i <= n ) THEN
              WRITE( out, 2050 ) i, VNAME( i ), state, prob%X( i ),            &
                   prob%X_l( i ), prob%X_u( i ), prob%Z( i )
            END IF
          END DO 
        END DO 

!  Compute the number of fixed and degenerate variables.

        nfixed = 0 ; ndegen = 0 
        DO i = 1, prob%n 
          IF ( ABS( prob%X_u( i ) - prob%X_l( i ) ) < stopr ) THEN
            nfixed = nfixed + 1 
            IF ( ABS( prob%Z( i ) ) < ten * stopr ) ndegen = ndegen + 1 
          ELSE IF ( MIN( ABS( prob%X( i ) - prob%X_l( i ) ),                   &
                    ABS( prob%X( i ) - prob%X_u( i ) ) ) <=                    &
                    MAX( ten * stopr, ABS( prob%Z( i ) ) ) ) THEN
            nfixed = nfixed + 1 
            IF ( ABS( prob%Z( i ) ) < ten * stopr ) ndegen = ndegen + 1 
          END IF 
        END DO 

!  Print details of the constraints.

        IF ( prob%m > 0 ) THEN 

          WRITE( out, 2040 ) 
          l = 2  ; IF ( fulsol ) l = prob%m 
          IF ( do_presolve ) THEN
            IF ( control%QPB_control%print_level >= DEBUG ) l = prob%m
          END IF
          DO j = 1, 2 
            IF ( j == 1 ) THEN 
              ir = 1 ; ic = MIN( l, prob%m ) 
            ELSE 
              IF ( ic < prob%m - l ) WRITE( out, 2000 ) 
              ir = MAX( ic + 1, prob%m - ic + 1 ) ; ic = prob%m 
            END IF 
            DO i = ir, ic 
              state = ' FREE' 
              IF ( ABS( prob%C( I ) - prob%C_l( i ) ) < ten * stopr )          &
                state = 'LOWER' 
              IF ( ABS( prob%C( I ) - prob%C_u( i ) ) < ten * stopr )          &
                state = 'UPPER' 
              IF ( ABS( prob%C_l( i ) - prob%C_u( i ) ) < stopr )              &
                state = 'EQUAL' 
              IF ( i <= m ) THEN
                WRITE( out, 2130 ) i, CNAME( i ), STATE, prob%C( i ),          &
                  prob%C_l( i ), prob%C_u( i ), prob%Y( i ) 
              END IF
            END DO 
          END DO 

!  Compute the number of equality, fixed inequality and degenerate constraints

          mequal = 0 ; mfixed = 0 ; mdegen = 0 
          DO i = 1, prob%m 
           IF ( ABS( prob%C_l( i ) - prob%C_u( i ) ) < stopr ) THEN
              mequal = mequal + 1
              IF ( ABS( prob%Y( i ) ) < stopr ) mdegen = mdegen + 1 
            ELSE IF ( MIN( ABS( prob%C( i ) - prob%C_l( i ) ),                 &
                      ABS( prob%C( i ) - prob%C_u( i ) ) ) <=                  &
                 MAX( ten * stopr, ABS( prob%Y( i ) ) ) ) THEN
              mfixed = mfixed + 1
              IF ( ABS( prob%Y( i ) ) < stopr ) mdegen = mdegen + 1 
            END IF
!           IF ( ABS( prob%C( i ) - prob%C_l( i ) ) < ten * stopr.OR.&
!                ABS( prob%C( i ) - prob%C_u( i ) ) < ten * stopr)THEN
!             IF ( ABS( prob%C_l( i )-prob%C_u( i ) )<ten * stopr)THEN
!                mequal = mequal + 1 
!             ELSE 
!                mfixed = mfixed + 1 
!             END IF 
!             IF ( ABS( prob%Y( i ) ) < stopr ) mdegen = mdegen + 1 
!           END IF 
          END DO 
        END IF 
        WRITE( out, 2100 ) prob%n, nfixed, ndegen 
        IF ( m > 0 ) THEN 
           WRITE( out, 2110 ) prob%m, mequal, mdegen 
           IF ( prob%m /= mequal ) WRITE( out, 2120 ) mfixed 
        END IF 
        WRITE( out, 2030 ) qfval, res_c, res_k, max_cs, iter, nfacts,          &
                           factorization_integer, factorization_real 

!  If required, write the solution to a file

        IF ( write_solution ) THEN
          INQUIRE( FILE = sfilename, EXIST = filexx )
          IF ( filexx ) THEN
             OPEN( sfiledevice, FILE = sfilename, FORM = 'FORMATTED',          &
                 STATUS = 'OLD', IOSTAT = iores )
          ELSE
             OPEN( sfiledevice, FILE = sfilename, FORM = 'FORMATTED',          &
                  STATUS = 'NEW', IOSTAT = iores )
          END IF
          IF ( iores /= 0 ) THEN 
            write( out, 2160 ) iores, sfilename
            STOP
          END IF

          WRITE( sfiledevice, 2250 ) pname, solv, qfval
          WRITE( sfiledevice, 2090 ) 

          DO i = 1, prob%n 
            state = ' FREE' 
            IF ( ABS( prob%X( i )   - prob%X_l( i ) ) < ten * stopr )          &
              state = 'LOWER' 
            IF ( ABS( prob%X( i )   - prob%X_u( i ) ) < ten * stopr )          &
              state = 'UPPER' 
            IF ( ABS( prob%X_l( I ) - prob%X_u( I ) ) < stopr )                &
              state = 'FIXED' 
            IF ( i <= n ) THEN
              WRITE( sfiledevice, 2050 ) i, VNAME( i ), STATE, prob%X( i ),    &
                prob%X_l( i ), prob%X_u( i ), prob%Z( i )
            END IF
          END DO 
  
          IF ( m > 0 ) THEN 
            WRITE( sfiledevice, 2040 ) 
            DO i = 1, prob%m 
              state = ' FREE' 
              IF ( ABS( prob%C( I ) - prob%C_l( i ) ) < ten * stopr )          &
                state = 'LOWER'
              IF ( ABS( prob%C( I ) - prob%C_u( i ) ) < ten * stopr )          &
                state = 'UPPER'
              IF ( ABS( prob%C_l( i ) - prob%C_u( i ) ) < stopr )              &
                state = 'EQUAL' 
              IF ( i <= m ) THEN
                WRITE( sfiledevice, 2130 ) i, CNAME(i), STATE, prob%C(i),      &
                  prob%C_l( i ), prob%C_u( i ), prob%Y( i )   
              END IF 
            END DO 
          END IF 
  
          WRITE( sfiledevice, 2030 ) qfval, res_c, res_k, max_cs, iter,        &
            nfacts, factorization_integer, factorization_real 
          CLOSE( sfiledevice ) 
        END IF 
      END IF 

      times = times - time ; timet = timet - timeo
      WRITE( out, 2060 ) times + timet 
      WRITE( out, 2070 ) pname 

!  Compare the variants used so far

      WRITE( out, 2080 ) solv, iter, nfacts, qfval, status, times, timet,      &
                         times + timet 

      IF ( write_result_summary ) THEN
        BACKSPACE( rfiledevice )
        WRITE( rfiledevice, 2190 )                                             &
           pname, n, m, iter, newton, nmods, qfval, status, timet
      END IF

!  Print the first and last few components of the solution.

!     WRITE( out, 2070 )
!     WRITE( out, 2120 )
!     j = MIN( npm, 12 )
!     RES( : n ) = prob%X( : n ) ; RES( np1 : npm ) = prob%Y( : m )
!     WRITE( out, 2100 ) ( RES( i ), i = 1, j )
!     IF ( j < npm ) THEN
!        IF ( j + 1 < npm - 11 ) WRITE( out, 2110 )
!        WRITE( out, 2100 ) ( RES( i ), i = MAX( j + 1, npm - 11 ), npm )
!     END IF
!     DEALLOCATE( RES )
      DEALLOCATE( VNAME, CNAME, C )

      CALL LPQPB_terminate( data, control, inform )
      CALL CUTEST_cterminate( cutest_status )
      STOP

 910  CONTINUE
      WRITE( out, "( ' CUTEst error, status = ', i0, ', stopping' )" )         &
        cutest_status
      STOP

!  Non-executable statements

 2000 FORMAT( '      . .          .....  ..........',                          &
              '  ..........  ..........  .......... ' ) 
 2010 FORMAT( /,' Stopping with inform%status = ', I3 ) 
 2020 FORMAT( /, ' Problem: ', A10 )
 2030 FORMAT( /, ' Final objective function value  ', ES22.14, /,              &
                 ' Maximum constraint violation    ', ES22.14, /,              &
                 ' Maximum dual infeasibility      ', ES22.14, /,              &
                 ' Maximum complementary slackness ', ES22.14, /,              &
          ' Total number of iterations = ',I6,' Number of factorizations = ',  &
          I6, //, I10, ' integer and ', I10, ' real words required',           &
          ' for the factorization' ) 
 2040 FORMAT( /,' Constraints : ', /, '                              ',        &
                '        <------ Bounds ------> ', /                           &
                '      # name       state    value   ',                        &
                '    Lower       Upper     Multiplier ' ) 
 2050 FORMAT( I7, 1X, A10, A6, 4ES12.4 ) 
 2060 FORMAT( /, ' Total time = ', 0P, F12.2 ) 
 2070 FORMAT( /, ' Problem: ', A10, //,                                        &
              '                                 objective',                    &
              '          < ------ time ----- > ', /,                           &
              ' Method   iterations  factors      value  ',                    &
              '   status setup   solve   total', /,                            &
              ' ------   ----------  -------    ---------',                    &
              '   ------ -----   -----   -----  ' ) 
 2080 FORMAT( 1X, A5, 2I10, 5X, ES12.4, I6, 0P, 3F8.2 ) 
 2090 FORMAT( /,' Solution : ', /,'                              ',           &
                '        <------ Bounds ------> ', /                          &
                '      # name       state    value   ',                       &
                '    Lower       Upper       Dual ' ) 
 2100 FORMAT( /, ' Of the ', I7, ' variables, ', 2X, I7,                      &
              ' are on bounds, &', I7, ' are dual degenerate' ) 
 2110 FORMAT( ' Of the ', I7, ' constraints, ', I7,' are equations, &', I7,   &
              ' are degenerate' ) 
 2120 FORMAT( ' Of the inequality constraints ', I6, ' are on bounds' ) 
 2130 FORMAT( I7, 1X, A10, A6, 4ES12.4 ) 
 2150 FORMAT( ' Allocation error, variable ', A8, ' status = ', I6 )
 2160 FORMAT( ' IOSTAT = ', I6, ' when opening file ', A9, '. Stopping ' )
 2190 FORMAT( A10, 2I7, I8, I6, ES13.4, I6, 0P, F8.1, 1X, A1, A1 ) 
 2250 FORMAT( /, ' Problem:    ', A10, /, ' Solver :   ', A5,                  &
              /, ' Objective:', ES24.16 )

!  End of program RUNLPQPB

      END PROGRAM RUNLPQPB


