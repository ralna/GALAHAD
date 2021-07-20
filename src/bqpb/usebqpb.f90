! THIS VERSION: GALAHAD 3.3 - 20/05/2021 AT 11:00 GMT.

!-*-*-*-*-*-*-*-  G A L A H A D   U S E B Q P B   M O D U L E  -*-*-*-*-*-*-*-*-

!  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
!  Principal authors: Nick Gould and Dominique Orban

!  History -
!   originally released with GALAHAD Version 2.4. January 18th 2010

!  For full documentation, see
!   http://galahad.rl.ac.uk/galahad-www/specs.html

    MODULE GALAHAD_USEBQPB_double

!    ---------------------------------------------------
!    | CUTEst/AMPL interface to BQPB, a preconditioned |
!    | conjugate-gradient interior-point algorithm for |
!    | bound-constrained convex quadratic programming  |
!    ---------------------------------------------------

      USE CUTEst_interface_double
!NOT95USE GALAHAD_CPU_time
      USE GALAHAD_QPT_double
      USE GALAHAD_SORT_double, only: SORT_reorder_by_rows
      USE GALAHAD_BQPB_double
      USE GALAHAD_SPECFILE_double 
      USE GALAHAD_COPYRIGHT
      USE GALAHAD_SCALING_double
      USE GALAHAD_SYMBOLS,                                                     &
          ACTIVE                => GALAHAD_ACTIVE,                             &
          TRACE                 => GALAHAD_TRACE,                              &
          DEBUG                 => GALAHAD_DEBUG,                              &
          GENERAL               => GALAHAD_GENERAL,                            &
          ALL_ZEROS             => GALAHAD_ALL_ZEROS

      IMPLICIT NONE

      PRIVATE
      PUBLIC :: USE_BQPB

    CONTAINS

!-*-*-*-*-*-*-*-*-*-   U S E _ B Q P B  S U B R O U T I N E   -*-*-*-*-*-*-*-*-

     SUBROUTINE USE_BQPB( input )

!  --------------------------------------------------------------------
!
!  Solve the quadratic program from CUTEst
!
!     minimize     1/2 x(T) H x + g(T) x
!
!     subject to       x_l <= x <= x_u
!
!  using the GALAHAD package GALAHAD_BQPB
!
!  --------------------------------------------------------------------

!  Dummy argument

      INTEGER, INTENT( IN ) :: input

!  Parameters

      INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
      REAL ( KIND = wp ), PARAMETER :: zero = 0.0_wp
      REAL ( KIND = wp ), PARAMETER :: one = 1.0_wp
      REAL ( KIND = wp ), PARAMETER :: ten = 10.0_wp
      REAL ( KIND = wp ), PARAMETER :: infinity = ten ** 19

!  Scalars

      INTEGER :: n, ir, ic, ifail, lh, liw, iores, smt_stat, cutest_status
      INTEGER :: i, j, l, neh, nfixed, ndegen, status, alloc_stat, H_ne, iter
!     INTEGER :: np1, npm
!     INTEGER :: factorization_integer, factorization_real
      REAL :: time, timeo, times, timet
      REAL ( KIND = wp ) :: objf, qfval, stopr, dummy
      REAL ( KIND = wp ) :: res_k, max_cs
      LOGICAL :: filexx, printo, printe, is_specfile
!     LOGICAL :: ldummy
            
!  Specfile characteristics

      INTEGER, PARAMETER :: input_specfile = 34
      INTEGER, PARAMETER :: lspec = 21
      CHARACTER ( LEN = 16 ) :: specname = 'RUNBQPB'
      TYPE ( SPECFILE_item_type ), DIMENSION( lspec ) :: spec
      CHARACTER ( LEN = 16 ) :: runspec = 'RUNBQPB.SPC'

!  The default values for BQPB could have been set as:

! BEGIN RUNBQPB SPECIFICATIONS (DEFAULT)
!  write-problem-data                                NO
!  problem-data-file-name                            BQPB.data
!  problem-data-file-device                          26
!  write-initial-sif                                 NO
!  initial-sif-file-name                             INITIAL.SIF
!  initial-sif-file-device                           51
!  least-squares-qp                                  NO
!  scale-problem                                     0
!  solve-problem                                     YES
!  print-full-solution                               NO
!  write-solution                                    NO
!  solution-file-name                                BQPBSOL.d
!  solution-file-device                              62
!  write-result-summary                              NO
!  result-summary-file-name                          BQPBRES.d
!  result-summary-file-device                        47
!  perturb-bounds-by                                 0.0
! END RUNBQPB SPECIFICATIONS

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
      CHARACTER ( LEN = 30 ) :: dfilename = 'BQPB.data'
      CHARACTER ( LEN = 30 ) :: ifilename = 'INITIAL.SIF'
      CHARACTER ( LEN = 30 ) :: rfilename = 'BQPBRES.d'
      CHARACTER ( LEN = 30 ) :: sfilename = 'BQPBSOL.d'
      LOGICAL :: do_solve = .TRUE.
      LOGICAL :: fulsol = .FALSE. 
      REAL ( KIND = wp ) :: pert_bnd = zero

!  Output file characteristics

      INTEGER, PARAMETER :: out  = 6
      INTEGER, PARAMETER :: io_buffer = 11
      INTEGER :: errout = 6
      CHARACTER ( LEN =  5 ) :: state, solv
      CHARACTER ( LEN = 10 ) :: pname

!  Arrays

      TYPE ( SCALING_control_type ) :: control
      TYPE ( BQPB_data_type ) :: data
      TYPE ( BQPB_control_type ) :: BQPB_control        
      TYPE ( BQPB_inform_type ) :: BQPB_inform
      TYPE ( QPT_problem_type ) :: prob

!  Allocatable arrays

      CHARACTER ( LEN = 10 ), ALLOCATABLE, DIMENSION( : ) :: VNAME
      REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: SH, SA
      REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: HX
      INTEGER, ALLOCATABLE, DIMENSION( : ) :: IW, X_stat

      CALL CPU_TIME( time )

!  Determine the number of variables and constraints

      CALL CUTEST_udimen( cutest_status, input, n )
      IF ( cutest_status /= 0 ) GO TO 910

!  Allocate suitable arrays

      ALLOCATE( prob%X( n ), prob%X_l( n ), prob%X_u( n ),                     &
                prob%G( n ), VNAME( n ), X_stat( n ), STAT = alloc_stat )
      IF ( alloc_stat /= 0 ) THEN
        WRITE( out, 2150 ) 'X', alloc_stat ; STOP
      END IF

!  Set up the data structures necessary to hold the group partially
!  separable function.

      CALL CUTEST_usetup( cutest_status, input, out, io_buffer,                &
                          n, prob%X, prob%X_l, prob%X_u )
      IF ( cutest_status /= 0 ) GO TO 910

!  Allocate derived types

      ALLOCATE( prob%X0( n ), STAT = alloc_stat )
      IF ( alloc_stat /= 0 ) THEN
        WRITE( out, 2150 ) 'X0', alloc_stat
        STOP
      END IF

!  Determine the names of the problem, variables and constraints.

      CALL CUTEST_unames( cutest_status, n, pname, VNAME )
      IF ( cutest_status /= 0 ) GO TO 910
      WRITE( out, 2020 ) pname 
!      WRITE( out, "( /, ' n   = ', i8, ' n = ', i8 )" ) n, n

!  Set up the initial estimate of the solution and
!  right-hand-side of the Kuhn-Tucker system.

!  Determine the constant terms for the problem functions.

      prob%X( : n ) = MIN( prob%X_u( : n ),                                    &
                           MAX( prob%X_l( : n ), prob%X( : n ) ) )

!  Set X0 to zero to determine the constant terms for the problem functions

      prob%X0 = zero 

!  Evaluate the constant term of the objective function

      CALL CUTEST_ufn( cutest_status, n, prob%X0, objf )
      IF ( cutest_status /= 0 ) GO TO 910

!  Allocate arrays to hold the Jacobian

      ALLOCATE( prob%A%row( 0 ), prob%A%col( 0 ), prob%A%val( 0 ),             &
                STAT = alloc_stat )
      IF ( alloc_stat /= 0 ) THEN
        WRITE( out, 2150 ) 'A', alloc_stat ; STOP
      END IF

!  Evaluate the gradient of the objective function

      CALL CUTEST_ugr( cutest_status, n, prob%X0, prob%G )
      IF ( cutest_status /= 0 ) GO TO 910

!  Determine the number of nonzeros in the Hessian

      CALL CUTEST_udimsh( cutest_status, lh )
      IF ( cutest_status /= 0 ) GO TO 910
      lh = MAX( lh, 1 )

!  Allocate arrays to hold the Hessian

      ALLOCATE( prob%H%row( lh ), prob%H%col( lh ), prob%H%val( lh ),          &
                STAT = alloc_stat )
      IF ( alloc_stat /= 0 ) THEN
        WRITE( out, 2150 ) 'H', alloc_stat
        STOP
      END IF

!  Evaluate the Hessian of the objective at the initial point.

      CALL CUTEST_ush( cutest_status, n, prob%X,                               &
                       neh, lh, prob%H%val, prob%H%row, prob%H%col )
      IF ( cutest_status /= 0 ) GO TO 910

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
      prob%Z( : n ) = one

      liw = n + 1
      ALLOCATE( prob%H%ptr( n + 1 ), prob%A%ptr( 0 ) )
      ALLOCATE( IW( liw ) )

!  Transform H to row storage format

      IF ( H_ne /= 0 ) THEN
        CALL SORT_reorder_by_rows( n, n, H_ne, prob%H%row, prob%H%col, H_ne,   &
                                   prob%H%val, prob%H%ptr, n + 1, IW, liw,     &
                                   out, out, i )
      ELSE
        prob%H%ptr = 0
      END IF

!  Deallocate arrays holding matrix row indices

      DEALLOCATE( prob%H%row )
      DEALLOCATE( IW )
      ALLOCATE( prob%H%row( 0 ) )

      prob%new_problem_structure = .TRUE.

!  Store the problem dimensions

      prob%n = n
      IF ( ALLOCATED( prob%H%type ) ) DEALLOCATE( prob%H%type )
      CALL SMT_put( prob%H%type, 'SPARSE_BY_ROWS', smt_stat )
      prob%f = objf
        
!  ------------------- problem set-up complete ----------------------

      CALL CPU_TIME( times )

!  ------------------ Open the specfile for runbqpb ----------------

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
        spec( 8 )%keyword = 'scale-problem'
        spec( 13 )%keyword = 'solve-problem'
        spec( 14 )%keyword = 'print-full-solution'
        spec( 15 )%keyword = 'write-solution'
        spec( 16 )%keyword = 'solution-file-name'
        spec( 17 )%keyword = 'solution-file-device'
        spec( 18 )%keyword = 'write-result-summary'
        spec( 19 )%keyword = 'result-summary-file-name'
        spec( 20 )%keyword = 'result-summary-file-device'
        spec( 21 )%keyword = 'perturb-bounds-by'

!   Read the specfile

        CALL SPECFILE_read( input_specfile, specname, spec, lspec, errout )

!   Interpret the result

        CALL SPECFILE_assign_logical( spec( 1 ), write_problem_data, errout )
        CALL SPECFILE_assign_string ( spec( 2 ), dfilename, errout )
        CALL SPECFILE_assign_integer( spec( 3 ), dfiledevice, errout )
        CALL SPECFILE_assign_logical( spec( 4 ), write_initial_sif, errout )
        CALL SPECFILE_assign_string ( spec( 5 ), ifilename, errout )
        CALL SPECFILE_assign_integer( spec( 6 ), ifiledevice, errout )
        CALL SPECFILE_assign_integer( spec( 8 ), scale, errout )
        CALL SPECFILE_assign_logical( spec( 13 ), do_solve, errout )
        CALL SPECFILE_assign_logical( spec( 14 ), fulsol, errout )
        CALL SPECFILE_assign_logical( spec( 15 ), write_solution, errout )
        CALL SPECFILE_assign_string ( spec( 16 ), sfilename, errout )
        CALL SPECFILE_assign_integer( spec( 17 ), sfiledevice, errout )
        CALL SPECFILE_assign_logical( spec( 18 ), write_result_summary, errout )
        CALL SPECFILE_assign_string ( spec( 19 ), rfilename, errout )
        CALL SPECFILE_assign_integer( spec( 20 ), rfiledevice, errout )
        CALL SPECFILE_assign_real( spec( 21 ), pert_bnd, errout )
      END IF

!  Perturb bounds if required

      IF ( pert_bnd /= zero ) THEN
        DO i = 1, n
          IF (  prob%X_l( i ) /= prob%X_u( i ) ) THEN
            IF ( prob%X_l( i ) > - infinity )                                  &
              prob%X_l( i ) = prob%X_l( i ) - pert_bnd
            IF ( prob%X_u( i ) < infinity )                                    &
              prob%X_u( i ) = prob%X_u( i ) + pert_bnd
          END IF 
        END DO
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

        WRITE( dfiledevice, "( 'n = ', 2I6, ' obj = ', ES12.4 )" ) n, prob%f
        WRITE( dfiledevice, "( ' g ', /, ( 5ES12.4 ) )" ) prob%G( : n )
        WRITE( dfiledevice, "( ' x_l ', /, ( 5ES12.4 ) )" ) prob%X_l( : n )
        WRITE( dfiledevice, "( ' x_u ', /, ( 5ES12.4 ) )" ) prob%X_u( : n )
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

!  Set all default values, and override defaults if requested
 
      CALL SCALING_initialize( control )

      CALL BQPB_initialize( data, BQPB_control, BQPB_inform )
      IF ( is_specfile )                                                       &
        CALL BQPB_read_specfile( BQPB_control, input_specfile )

      control%print_level = BQPB_control%print_level
      control%out = BQPB_control%out
      control%out_error = BQPB_control%error

      printo = out > 0 .AND. control%print_level > 0
      printe = out > 0 .AND. control%print_level >= 0
      WRITE( out, 2200 ) n, H_ne

      IF ( printo ) CALL COPYRIGHT( out, '2010' )
      X_stat = 0

!  If required, scale the problem

      IF ( scale > 0 ) THEN
        ALLOCATE( SH( n ), SA( 0 ), STAT = alloc_stat )
        IF ( alloc_stat /= 0 ) THEN
          IF ( printe ) WRITE( out, 2150 ) 'SH/SA', alloc_stat ; STOP
        END IF

!  Scale using K

        IF ( scale == 1 .OR. scale == 4 ) THEN
          IF ( printo ) WRITE( out, 2140 ) 'K'
          CALL SCALING_get_factors_from_K( n, 0, prob%H%val, prob%H%col,       &
                                           prob%H%ptr, prob%A%val, prob%A%col, &
                                           prob%A%ptr, SH, SA, control, ifail )
!  Scale using A

        ELSE IF ( scale == 2 .OR. scale == 5 ) THEN
          IF ( printo ) WRITE( out, 2140 ) 'A'
          CALL SCALING_get_factors_from_A( n, 0, prob%A%val, prob%A%col,       &
                                           prob%A%ptr, SH, SA, control, ifail )
        ELSE IF ( scale == 3 ) THEN
          SH = one ; SA = one
        END IF

!  Rescale A

        IF ( scale >= 3 ) THEN
          IF ( printo ) WRITE( out, 2170 )
          CALL SCALING_normalize_rows_of_A( n, 0, prob%A%val, prob%A%col,      &
                                            prob%A%ptr, SH, SA )
        END IF

!  Apply the scaling factors

        CALL SCALING_apply_factors( n, 0, prob%H%val, prob%H%col, prob%H%ptr,  &
                                    prob%A%val, prob%A%col, prob%A%ptr,        &
                                    prob%G, prob%X, prob%X_l, prob%X_u,        &
                                    prob%C_l, prob%C_u, prob%Y, prob%Z,        &
                                    infinity, SH, SA, .TRUE. )
      END IF

!  If the preprocessor is to be used, or the problem to be output,
!  allocate sufficient space

      IF ( write_initial_sif ) THEN

        ALLOCATE( prob%X_status( n ), STAT = alloc_stat )
        IF ( alloc_stat /= 0 ) THEN
          IF ( printe ) WRITE( out, 2150 ) 'X_status', alloc_stat
          STOP
        END IF
        prob%X_status = ACTIVE
        
        ALLOCATE( prob%Z_l( n ), prob%Z_u( n ), STAT = alloc_stat )
        IF ( alloc_stat /= 0 ) THEN
          IF ( printe ) WRITE( out, 2150 ) 'Z_lu', alloc_stat
          STOP
        END IF
        prob%Z_l( : n ) = - infinity
        prob%Z_u( : n ) =   infinity
        
!  Writes the initial SIF file, if needed

        IF ( write_initial_sif ) THEN
          CALL QPT_write_to_sif( prob, pname, ifilename, ifiledevice,          &
                                 .FALSE., .FALSE., infinity )
          IF ( .NOT. do_solve ) STOP
        END IF
      END IF

!  Call the optimizer

      qfval = objf 

      IF ( do_solve .AND. prob%n > 0 ) THEN

        CALL CPU_TIME( timeo )
  
        prob%n = n
        DEALLOCATE( prob%X0 )
    
        solv = ' BQPB'
        IF ( printo ) WRITE( out, " ( ' ** BQPB solver used ** ' ) " )
        CALL BQPB_solve( prob, data, BQPB_control, BQPB_inform,                &
                         X_stat = X_stat )

        IF ( printo ) WRITE( out, " ( /, ' ** BQPB solver used ** ' ) " )
        qfval = BQPB_inform%obj
  
        CALL CPU_TIME( timet )
  
!  Deallocate arrays from the minimization
  
        status = BQPB_inform%status
        iter = BQPB_inform%iter
        stopr = BQPB_control%stop_abs_d
!       factorization_integer = BQPB_inform%factorization_integer 
!       factorization_real = BQPB_inform%factorization_real
        CALL BQPB_terminate( data, BQPB_control, BQPB_inform )
      ELSE
        timeo  = 0.0
        timet  = 0.0
        iter   = 0
        solv   = ' NONE'
        status = 0
        stopr = BQPB_control%stop_abs_d
!       factorization_integer = 0 ; factorization_real    = 0
        qfval  = prob%f
      END IF

!  If the problem was scaled, unscale it.

      IF ( scale > 0 ) THEN
        CALL SCALING_apply_factors( n, 0, prob%H%val, prob%H%col, prob%H%ptr,  &
                                    prob%A%val, prob%A%col, prob%A%ptr,        &
                                    prob%G, prob%X, prob%X_l, prob%X_u,        &
                                    prob%C_l, prob%C_u, prob%Y, prob%Z,        &
                                    infinity, SH, SA, .FALSE., C = prob%C )
        DEALLOCATE( SH, SA )
      END IF

!  Compute maximum complementary slackness

      max_cs = zero
      DO i = 1, n
        dummy = prob%X( i )
        IF ( prob%X_l( i ) > - infinity ) THEN
          IF ( prob%X_u( i ) < infinity ) THEN
            max_cs = MAX( max_cs,                                              &
                 MIN( ABS( ( prob%X_l( i ) - dummy ) * prob%Z( i ) ),          &
                      ABS( ( prob%X_u( i ) - dummy ) * prob%Z( i ) ) ) )
          ELSE
            max_cs = MAX( max_cs,                                              &
                          ABS( ( prob%X_l( i ) - dummy ) * prob%Z( i ) ) )
          END IF
        ELSE IF ( prob%X_u( i ) < infinity ) THEN
          max_cs = MAX( max_cs, ABS( ( prob%X_u( i ) - dummy ) * prob%Z( i ) ) )
        END IF
      END DO

!  Compute maximum KKT residual

      ALLOCATE( HX( n ), STAT = alloc_stat )
      HX = prob%G( : n )
      DO i = 1, n
        DO l = prob%H%ptr( i ), prob%H%ptr( i + 1 ) - 1
          j = prob%H%col( l )
          HX( i ) = HX( i ) + prob%H%val( l ) * prob%X( j )
          IF ( j /= i )                                                        &
            HX( j ) = HX( j ) + prob%H%val( l ) * prob%X( i )
        END DO
      END DO
      res_k = MAXVAL( ABS( HX( : n ) - prob%Z( : n ) ) ) 

!  Print details of the solution obtained

      WRITE( out, 2010 ) status
      IF ( status == 0 .OR. status == - 8 .OR. status == - 9 .OR.              &
           status == - 10 ) THEN
        l = 4
        IF ( fulsol ) l = n 

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
            IF ( ABS( prob%X  ( i ) - prob%X_l( i ) ) < ten * stopr )          &
              state = 'LOWER'
            IF ( ABS( prob%X  ( i ) - prob%X_u( i ) ) < ten * stopr )          &
              state = 'UPPER'
            IF ( ABS( prob%X_l( i ) - prob%X_u( i ) ) <     1.0D-10 )          &
              state = 'FIXED'
            WRITE( out, 2050 ) i, VNAME( i ), state, prob%X( i ),              &
                               prob%X_l( i ), prob%X_u( i ), prob%Z( i )
          END DO 
        END DO 

!  Compute the number of fixed and degenerate variables.

        nfixed = 0 ; ndegen = 0 
        DO i = 1, n 
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

        WRITE( out, 2100 ) n, nfixed, ndegen 
        WRITE( out, 2030 ) qfval, res_k, max_cs, iter
!                          factorization_integer, factorization_real 

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

          DO i = 1, n 
            state = ' FREE' 
            IF ( ABS( prob%X( i )   - prob%X_l( i ) ) < ten * stopr )          &
              state = 'LOWER' 
            IF ( ABS( prob%X( i )   - prob%X_u( i ) ) < ten * stopr )          &
              state = 'UPPER' 
            IF ( ABS( prob%X_l( I ) - prob%X_u( I ) ) < stopr )                &
              state = 'FIXED' 
            WRITE( sfiledevice, 2050 ) i, VNAME( i ), STATE, prob%X( i ),      &
              prob%X_l( i ), prob%X_u( i ), prob%Z( i )
          END DO 
  
          WRITE( sfiledevice, 2030 ) qfval, res_k, max_cs, iter
!           factorization_integer, factorization_real 
          CLOSE( sfiledevice ) 
        END IF 
      END IF 

      times = times - time ; timet = timet - timeo
      WRITE( out, 2060 ) times + timet 
      WRITE( out, 2070 ) pname 

!  Compare the variants used so far

      WRITE( out, 2080 ) solv, iter, qfval, status, times,                     &
                         timet, times + timet 

      IF ( write_result_summary ) THEN
        BACKSPACE( rfiledevice )
        WRITE( rfiledevice, 2190 )                                             &
           pname, n, iter, qfval, status, timet
      END IF

      DEALLOCATE( VNAME )
      IF ( is_specfile ) CLOSE( input_specfile )

      CALL CUTEST_cterminate( cutest_status )
      RETURN

 910  CONTINUE
      WRITE( out, "( ' CUTEst error, status = ', i0, ', stopping' )" )         &
        cutest_status
      status = - 98
      RETURN

!  Non-executable statements

 2000 FORMAT( '      . .          .....  ..........',                          &
              '  ..........  ..........  .......... ' ) 
 2010 FORMAT( /,' Stopping with inform%status = ', I0 ) 
 2020 FORMAT( /, ' Problem: ', A10 )
 2030 FORMAT( /, ' Final objective function value  ', ES22.14, /,              &
                 ' Maximum dual infeasibility      ', ES22.14, /,              &
                 ' Maximum complementary slackness ', ES22.14, //,             &
                 ' Number of iterations = ', I0 )
!           //, I0, ' integer and ', I0, ' real words required',               &
!                ' for the factorization' ) 
 2050 FORMAT( I7, 1X, A10, A6, 4ES12.4 ) 
 2060 FORMAT( /, ' Total time = ', F0.2 ) 
 2070 FORMAT( /, ' Problem: ', A10, //,                                        &
                 '                     objective',                             &
                 '          < ------ time ----- > ', /,                        &
                 ' Method  iterations    value  ',                             &
                 '   status setup   solve   total', /,                         &
                 ' ------  ----------   -------   ',                           &
                 ' ------ -----    ----   -----  ' ) 
 2080 FORMAT( A5, I7, 6X, ES12.4, I6, 0P, 3F8.2 ) 
 2090 FORMAT( /, ' Solution : ', /, '                              ',          &
                 '        <------ Bounds ------> ', /                          &
                 '      # name       state    value   ',                       &
                 '    Lower       Upper       Dual ' ) 
 2100 FORMAT( /, ' Of the ', I0, ' variables, ', I0,                           &
              ' are on bounds & ', I0, ' are dual degenerate' ) 
! 2110 FORMAT( ' Of the ', I7, ' constraints, ', I7,' are equations &', I7,    &
!              ' are degenerate' ) 
 2140 FORMAT( /, ' *** Problem will be scaled based on ', A1, ' *** ' )
 2150 FORMAT( ' Allocation error, variable ', A8, ' status = ', I6 )
 2160 FORMAT( ' IOSTAT = ', I6, ' when opening file ', A9, '. Stopping ' )
 2170 FORMAT( /, ' *** Further scaling applied to A *** ' )
 2180 FORMAT( A10 )
 2190 FORMAT( A10, I7, I6, ES13.4, I6, 0P, F8.2 ) 
 2200 FORMAT( /, ' problem dimensions: n = ', I0, ', h_ne = ', I0 )
 2250 FORMAT( /, ' Problem:    ', A10, /, ' Solver :   ', A5,                  &
              /, ' Objective:', ES24.16 )

!  End of subroutine USE_BQPB

     END SUBROUTINE USE_BQPB

!  End of module USEBQPB_double

   END MODULE GALAHAD_USEBQPB_double


