! THIS VERSION: GALAHAD 3.3 - 27/01/2020 AT 10:30 GMT.

!-*-*-*-*-*-*-*-  G A L A H A D   U S E C Q P   M O D U L E  -*-*-*-*-*-*-*-*-

!  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
!  Principal authors: Nick Gould and Dominique Orban

!  History -
!   originally released with GALAHAD Version 2.4. January 1st 2010

!  For full documentation, see
!   http://galahad.rl.ac.uk/galahad-www/specs.html

    MODULE GALAHAD_USELPQP_double

!    -------------------------------------------------------------
!    | CUTEst/AMPL interface to LPQP, a program to               |
!    | assemble an l_p QP from an input quadratic program        |
!    -------------------------------------------------------------

      USE CUTEst_interface_double
!NOT95USE GALAHAD_CPU_time
      USE GALAHAD_QPT_double
      USE GALAHAD_LPQP_double
      USE GALAHAD_SORT_double, only: SORT_reorder_by_rows
      USE GALAHAD_SPECFILE_double
      USE GALAHAD_COPYRIGHT
      USE GALAHAD_SYMBOLS,                                                     &
          ACTIVE                => GALAHAD_ACTIVE,                             &
          TRACE                 => GALAHAD_TRACE,                              &
          DEBUG                 => GALAHAD_DEBUG,                              &
          GENERAL               => GALAHAD_GENERAL,                            &
          ALL_ZEROS             => GALAHAD_ALL_ZEROS
     USE GALAHAD_SMT_double, ONLY: SMT_put, SMT_get

      PRIVATE
      PUBLIC :: USE_LPQP

    CONTAINS

!-*-*-*-*-*-*-*-*-*-   U S E _ L P Q P  S U B R O U T I N E   -*-*-*-*-*-*-*-

     SUBROUTINE USE_LPQP( input, close_input )

!  --------------------------------------------------------------------
!
!      Formulate the l_1 quadratic program
!
!        minimize     1/2 x(T) H x + g(T) x + f
!      x_l<=x<=x_u        + rho_g min( A x - c_l , c_u - A x, 0 )
!
!      using the GALAHAD package LPQP
!
!  --------------------------------------------------------------------

!  Dummy argument

      INTEGER, INTENT( IN ) :: input
      LOGICAL, OPTIONAL, INTENT( IN ) :: close_input

!  Parameters

      INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
      REAL ( KIND = wp ), PARAMETER :: zero = 0.0_wp
      REAL ( KIND = wp ), PARAMETER :: one = 1.0_wp
      REAL ( KIND = wp ), PARAMETER :: ten = 10.0_wp
      REAL ( KIND = wp ), PARAMETER :: infinity = ten ** 19

!     INTEGER, PARAMETER :: n_k = 100, k_k = 3, in = 28
!     REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( :, : ) :: k_val
!     CHARACTER ( len = 10 ) :: filename = 'k.val'

!  Scalars

      INTEGER :: n, m, la, lh, liw, iores
      INTEGER :: i, j, l, neh, nea
      INTEGER :: alloc_stat, cutest_status, A_ne, H_ne
      REAL :: time, timel1, timel2, times
      REAL ( KIND = wp ) :: obj
      LOGICAL :: filexx, printo, printe

!  Specfile characteristics

      INTEGER, PARAMETER :: input_specfile = 34
      INTEGER, PARAMETER :: lspec = 10
      CHARACTER ( LEN = 16 ) :: specname = 'RUNLPQP'
      TYPE ( SPECFILE_item_type ), DIMENSION( lspec ) :: spec
      CHARACTER ( LEN = 16 ) :: runspec = 'RUNLPQP.SPC'

!  Default values for specfile-defined parameters

      INTEGER :: dfiledevice = 26
      INTEGER :: ifiledevice = 51
!     LOGICAL :: write_problem_data   = .FALSE.
      LOGICAL :: write_problem_data   = .TRUE.
      LOGICAL :: write_initial_sif    = .FALSE.
      CHARACTER ( LEN = 30 ) :: dfilename = 'LPQP.data'
      CHARACTER ( LEN = 30 ) :: ifilename = 'ORIG.SIF'
      CHARACTER ( LEN = 30 ) :: input_format = 'SPARSE_BY_ROWS'
      CHARACTER ( LEN = 30 ) :: output_format = 'SPARSE_BY_ROWS'
      REAL ( KIND = wp ) :: rho_g = 100000.0
      LOGICAL :: one_norm = .FALSE.

!  Output file characteristics

      INTEGER, PARAMETER :: out  = 6
      INTEGER, PARAMETER :: io_buffer = 11
      INTEGER :: errout = 6
      CHARACTER ( LEN = 10 ) :: pname

!  Arrays

      TYPE ( LPQP_control_type ) :: lpqp_control
      TYPE ( LPQP_inform_type ) :: lpqp_inform
      TYPE ( LPQP_data_type ) :: lpqp_data
      TYPE ( QPT_problem_type ) :: prob

!  Allocatable arrays

      CHARACTER ( LEN = 10 ), ALLOCATABLE, DIMENSION( : ) :: VNAME, CNAME
      CHARACTER ( LEN = 10 ), ALLOCATABLE, DIMENSION( : ) :: VNAME_lpqp,       &
                                                             CNAME_lpqp
      REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: X0, C
      LOGICAL, ALLOCATABLE, DIMENSION( : ) :: EQUATN, LINEAR
      INTEGER, ALLOCATABLE, DIMENSION( : ) :: IW

      CALL CPU_TIME( time )

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
        WRITE( out, "( ' nea = ', I0, ', la = ', I0 )" ) nea, la
        WRITE( out, 2150 ) 'H', alloc_stat
        STOP
      END IF

!  Evaluate the Hessian of the Lagrangian function at the initial point.

      CALL CUTEST_csh( cutest_status, n, m, prob%X, prob%Y,                    &
                       neh, lh, prob%H%val, prob%H%row, prob%H%col )
      IF ( cutest_status /= 0 ) GO TO 910
      WRITE( out, "( ' nea = ', I0, ', la = ', I0,                             &
     &               ', neh = ', I0, ', lh = ', I0 )" ) nea, la, neh, lh

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

!  ------------------ Open the specfile for runlpqp ----------------

      OPEN( input_specfile, FILE = runspec, FORM = 'FORMATTED', STATUS = 'OLD' )

!   Define the keywords

      spec( 1 )%keyword = 'write-problem-data'
      spec( 2 )%keyword = 'problem-data-file-name'
      spec( 3 )%keyword = 'problem-data-file-device'
      spec( 4 )%keyword = 'write-initial-sif'
      spec( 5 )%keyword = 'initial-sif-file-name'
      spec( 6 )%keyword = 'initial-sif-file-device'
      spec( 7 )%keyword = 'rho-used'
      spec( 8 )%keyword = 'input-format'
      spec( 9 )%keyword = 'output-format'
      spec( 10 )%keyword = 'one-norm-penalty'

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
      CALL SPECFILE_assign_string( spec( 8 ), input_format, errout )
      CALL SPECFILE_assign_string( spec( 9 ), output_format, errout )
      CALL SPECFILE_assign_logical( spec( 10 ), one_norm, errout )

!  for row input format

      IF ( TRIM( input_format ) == 'SPARSE_BY_ROWS' ) THEN

!  Transform A to row storage format

        IF ( A_ne /= 0 ) THEN
          CALL SORT_reorder_by_rows( m, n, A_ne, prob%A%row, prob%A%col, A_ne, &
                                     prob%A%val, prob%A%ptr, m + 1, IW, liw,   &
                                     out, out, i )
        ELSE
          prob%A%ptr = 1
        END IF

!  Same for H

        IF ( H_ne /= 0 ) THEN
          CALL SORT_REORDER_by_rows( n, n, H_ne, prob%H%row, prob%H%col, H_ne, &
                                     prob%H%val, prob%H%ptr, n + 1, IW, liw,   &
                                     out, out, i )
        ELSE
          prob%H%ptr = 1
        END IF

!  Deallocate arrays holding matrix row indices

        DEALLOCATE( prob%A%row, prob%H%row )
        DEALLOCATE( IW )
        ALLOCATE( prob%A%row( 0 ), prob%H%row( 0 ) )
        prob%A%ne = - 1
        prob%H%ne = - 1
        IF ( ALLOCATED( prob%A%type ) ) DEALLOCATE( prob%A%type )
        CALL SMT_put( prob%A%type, 'SPARSE_BY_ROWS', i )
        IF ( ALLOCATED( prob%H%type ) ) DEALLOCATE( prob%H%type )
        CALL SMT_put( prob%H%type, 'SPARSE_BY_ROWS', i )
      ELSE
        prob%A%ne = A_ne
        prob%H%ne = H_ne
        IF ( ALLOCATED( prob%A%type ) ) DEALLOCATE( prob%A%type )
        CALL SMT_put( prob%A%type, 'COORDINATE', i )
        IF ( ALLOCATED( prob%H%type ) ) DEALLOCATE( prob%H%type )
        CALL SMT_put( prob%H%type, 'COORDINATE', i )
      END IF

!  Store the problem dimensions

      prob%n = n
      prob%m = m
      prob%f = obj
      CLOSE( input  )

!  ------------------- problem set-up complete ----------------------

      CALL CPU_TIME( times )

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

        WRITE( dfiledevice, "( ' original formulation' )" )
        WRITE( dfiledevice, "( 'n, m = ', 2I8, ' obj = ', ES12.4 )" )          &
          n, m, prob%f
        WRITE( dfiledevice, "( ' g ', /, ( 5ES12.4 ) )" ) prob%G( : n )
        WRITE( dfiledevice, "( ' x_l ', /, ( 5ES12.4 ) )" ) prob%X_l( : n )
        WRITE( dfiledevice, "( ' x_u ', /, ( 5ES12.4 ) )" ) prob%X_u( : n )
        WRITE( dfiledevice, "( ' c_l ', /, ( 5ES12.4 ) )" ) prob%C_l( : m )
        WRITE( dfiledevice, "( ' c_u ', /, ( 5ES12.4 ) )" ) prob%C_u( : m )
        IF ( TRIM( input_format ) == 'COORDINATE' ) THEN
          A_ne = prob%A%ne
          H_ne = prob%H%ne
          WRITE( dfiledevice, "( ' A_row ', /, ( 10I6 ) )" )                   &
            prob%A%row( : A_ne )
          WRITE( dfiledevice, "( ' A_col ', /, ( 10I6 ) )" )                   &
            prob%A%col( : A_ne )
          WRITE( dfiledevice, "( ' A_val ', /, ( 5ES12.4 ) )" )                &
            prob%A%val( : A_ne )
          WRITE( dfiledevice, "( ' H_row ', /, ( 10I6 ) )" )                   &
            prob%H%row( : H_ne )
          WRITE( dfiledevice, "( ' H_col ', /, ( 10I6 ) )" )                   &
            prob%H%col( : H_ne )
          WRITE( dfiledevice, "( ' H_val ', /, ( 5ES12.4 ) )" )                &
            prob%H%val( : H_ne )
        ELSE
          A_ne = prob%A%ptr( prob%m + 1 ) - 1
          H_ne = prob%H%ptr( prob%n + 1 ) - 1
          WRITE( dfiledevice, "( ' A_ptr ', /, ( 10I6 ) )" )                   &
            prob%A%ptr( : prob%m + 1 )
          WRITE( dfiledevice, "( ' A_col ', /, ( 10I6 ) )" )                   &
            prob%A%col( : A_ne )
          WRITE( dfiledevice, "( ' A_val ', /, ( 5ES12.4 ) )" )                &
            prob%A%val( : A_ne )
          WRITE( dfiledevice, "( ' H_ptr ', /, ( 10I6 ) )" )                   &
            prob%H%ptr( : prob%n + 1 )
          WRITE( dfiledevice, "( ' H_col ', /, ( 10I6 ) )" )                   &
            prob%H%col( : H_ne )
          WRITE( dfiledevice, "( ' H_val ', /, ( 5ES12.4 ) )" )                &
            prob%H%val( : H_ne )
        END IF
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

!     CALL LPQP_initialize( lpqp_data, lpqp_control,                           &
      CALL LPQP_initialize( lpqp_control )
      CALL LPQP_read_specfile( lpqp_control, input_specfile )
      IF ( TRIM( output_format ) == 'COORDINATE' ) THEN
        lpqp_control%h_output_format = 'COORDINATE'
        lpqp_control%a_output_format = 'COORDINATE'
      END IF

      printo = out > 0 .AND. lpqp_control%print_level > 0
      printe = out > 0 .AND. lpqp_control%print_level >= 0

      IF ( printo ) CALL COPYRIGHT( out, '2010' )

!  Reformulate the problem
!  =======================

      CALL CPU_TIME( timel1 )

      CALL LPQP_formulate( prob, rho_g, one_norm, lpqp_data,                   &
                           lpqp_control, lpqp_inform,                          &
                           VNAME_lpqp = VNAME_lpqp, CNAME_lpqp = CNAME_lpqp )
      CALL CPU_TIME( timel2 )
      IF ( lpqp_inform%status /= 0 ) THEN
          WRITE( out, "( '  ERROR return from LPQP_formulate (status =', I6,   &
        &   ')' )" ) lpqp_inform%status
          STOP
        END IF

!  If required, print out the reformulated problem data

      IF ( write_problem_data ) THEN
        WRITE( dfiledevice, "( /, ' penalty reformulation' )" )
        WRITE( dfiledevice, "( 'n, m = ', 2I8, ' obj = ', ES12.4 )" )          &
          prob%n, prob%m, prob%f
        WRITE( dfiledevice, "( ' g ', /, ( 5ES12.4 ) )" ) prob%G( : prob%n )
        WRITE( dfiledevice, "( ' x_l ', /, ( 5ES12.4 ) )" ) prob%X_l( : prob%n )
        WRITE( dfiledevice, "( ' x_u ', /, ( 5ES12.4 ) )" ) prob%X_u( : prob%n )
        WRITE( dfiledevice, "( ' c_l ', /, ( 5ES12.4 ) )" ) prob%C_l( : prob%m )
        WRITE( dfiledevice, "( ' c_u ', /, ( 5ES12.4 ) )" ) prob%C_u( : prob%m )
        IF ( TRIM( output_format ) == 'COORDINATE' ) THEN
          a_ne = prob%A%ne
          h_ne = prob%H%ne
          WRITE( dfiledevice, "( ' A_row ', /, ( 10I6 ) )" )                   &
            prob%A%row( : A_ne )
          WRITE( dfiledevice, "( ' A_col ', /, ( 10I6 ) )" )                   &
            prob%A%col( : A_ne )
          WRITE( dfiledevice, "( ' A_val ', /, ( 5ES12.4 ) )" )                &
            prob%A%val( : A_ne )
          WRITE( dfiledevice, "( ' H_row ', /, ( 10I6 ) )" )                   &
            prob%H%row( : H_ne )
          WRITE( dfiledevice, "( ' H_col ', /, ( 10I6 ) )" )                   &
            prob%H%col( : H_ne )
          WRITE( dfiledevice, "( ' H_val ', /, ( 5ES12.4 ) )" )                &
            prob%H%val( : H_ne )
        ELSE
          a_ne = prob%A%ptr( prob%m + 1 ) - 1
          h_ne = prob%H%ptr( prob%n + 1 ) - 1
          WRITE( dfiledevice, "( ' A_ptr ', /, ( 10I6 ) )" )                   &
            prob%A%ptr( : prob%m + 1 )
          WRITE( dfiledevice, "( ' A_col ', /, ( 10I6 ) )" )                   &
            prob%A%col( : A_ne )
          WRITE( dfiledevice, "( ' A_val ', /, ( 5ES12.4 ) )" )                &
            prob%A%val( : A_ne )
          WRITE( dfiledevice, "( ' H_ptr ', /, ( 10I6 ) )" )                   &
            prob%H%ptr( : prob%n + 1 )
          WRITE( dfiledevice, "( ' H_col ', /, ( 10I6 ) )" )                   &
            prob%H%col( : H_ne )
          WRITE( dfiledevice, "( ' H_val ', /, ( 5ES12.4 ) )" )                &
            prob%H%val( : H_ne )
        END IF
        WRITE( dfiledevice, "( ' VNAME ', /, ( 8A10 ) )" )                     &
          VNAME_lpqp( : prob%n )
        WRITE( dfiledevice, "( ' CNAME ', /, ( 8A10 ) )" )                     &
          CNAME_lpqp( : prob%m )
      END IF

      IF ( printo ) WRITE( out, 2310 ) prob%n, prob%m, A_ne, H_ne,             &
         timel2 - timel1

      CALL LPQP_restore( prob, lpqp_data )

!  If required, print out the restored problem data

      IF ( write_problem_data ) THEN
        WRITE( dfiledevice, "( /, ' restored reformulation' )" )
        WRITE( dfiledevice, "( 'n, m = ', 2I8, ' obj = ', ES12.4 )" )          &
          prob%n, prob%m, prob%f
        WRITE( dfiledevice, "( ' g ', /, ( 5ES12.4 ) )" ) prob%G( : prob%n )
        WRITE( dfiledevice, "( ' x_l ', /, ( 5ES12.4 ) )" ) prob%X_l( : prob%n )
        WRITE( dfiledevice, "( ' x_u ', /, ( 5ES12.4 ) )" ) prob%X_u( : prob%n )
        WRITE( dfiledevice, "( ' c_l ', /, ( 5ES12.4 ) )" ) prob%C_l( : prob%m )
        WRITE( dfiledevice, "( ' c_u ', /, ( 5ES12.4 ) )" ) prob%C_u( : prob%m )
        IF ( TRIM( input_format ) == 'COORDINATE' ) THEN
          a_ne = prob%A%ne
          h_ne = prob%H%ne
          WRITE( dfiledevice, "( ' A_row ', /, ( 10I6 ) )" )                   &
            prob%A%row( : A_ne )
          WRITE( dfiledevice, "( ' A_col ', /, ( 10I6 ) )" )                   &
            prob%A%col( : A_ne )
          WRITE( dfiledevice, "( ' A_val ', /, ( 5ES12.4 ) )" )                &
            prob%A%val( : A_ne )
          WRITE( dfiledevice, "( ' H_row ', /, ( 10I6 ) )" )                   &
            prob%H%row( : H_ne )
          WRITE( dfiledevice, "( ' H_col ', /, ( 10I6 ) )" )                   &
            prob%H%col( : H_ne )
          WRITE( dfiledevice, "( ' H_val ', /, ( 5ES12.4 ) )" )                &
            prob%H%val( : H_ne )
        ELSE
          a_ne = prob%A%ptr( prob%m + 1 ) - 1
          h_ne = prob%H%ptr( prob%n + 1 ) - 1
          WRITE( dfiledevice, "( ' A_ptr ', /, ( 10I6 ) )" )                   &
            prob%A%ptr( : prob%m + 1 )
          WRITE( dfiledevice, "( ' A_col ', /, ( 10I6 ) )" )                   &
            prob%A%col( : A_ne )
          WRITE( dfiledevice, "( ' A_val ', /, ( 5ES12.4 ) )" )                &
            prob%A%val( : A_ne )
          WRITE( dfiledevice, "( ' H_ptr ', /, ( 10I6 ) )" )                   &
            prob%H%ptr( : prob%n + 1 )
          WRITE( dfiledevice, "( ' H_col ', /, ( 10I6 ) )" )                   &
            prob%H%col( : H_ne )
          WRITE( dfiledevice, "( ' H_val ', /, ( 5ES12.4 ) )" )                &
            prob%H%val( : H_ne )
        END IF
        CLOSE( dfiledevice )
      END IF

      DEALLOCATE( prob%X, prob%X_l, prob%X_u, prob%G, VNAME,                   &
                  prob%C_l, prob%C_u, prob%Y, prob%Z, CNAME,                   &
                  VNAME_LPQP, CNAME_LPQP, EQUATN,                              &
                  prob%C, prob%A%row, prob%A%col, prob%A%val, prob%A%ptr,      &
                  prob%H%row, prob%H%col, prob%H%val, prob%H%ptr,              &
                  prob%A%type, prob%H%type, C, STAT = alloc_stat )

      CALL LPQP_terminate( lpqp_data, lpqp_control, lpqp_inform )
      CALL CUTEST_cterminate( cutest_status )
      GO TO 920

 910  CONTINUE
      WRITE( out, "( ' CUTEst error, status = ', i0, ', stopping' )" )         &
        cutest_status

!  close the input file if required

 920  CONTINUE
      IF ( PRESENT( close_input ) ) THEN
        IF ( close_input ) THEN
          CLOSE( input )
          STOP
        END IF
      END IF
      RETURN

!  Non-executable statements

 2020 FORMAT( /, ' Problem: ', A )
 2150 FORMAT( ' Allocation error, variable ', A8, ' status = ', I0 )
 2160 FORMAT( ' IOSTAT = ', I0, ' when opening file ', A, '. Stopping ' )
 2310 FORMAT( ' dimensions of lpqp: n = ', I0, ', m = ', I0,                   &
              ', a_ne = ', I0, ', h_ne = ', I0, /,                             &
              ' formulating time = ', F0.2 )

!  End of subroutine USE_LPQP

     END SUBROUTINE USE_LPQP

!  End of module USELPQP_double

   END MODULE GALAHAD_USELPQP_double
