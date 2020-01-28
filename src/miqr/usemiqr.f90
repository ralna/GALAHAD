! THIS VERSION: GALAHAD 3.3 - 27/01/2020 AT 10:30 GMT.

!-*-*-*-*-*-*-*-  G A L A H A D   U S E M I Q R   M O D U L E  -*-*-*-*-*-*-*-*-

!  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
!  Principal authors: Nick Gould and Dominique Orban

!  History -
!   originally released with GALAHAD Version 2.6. May 23rd 2014

!  For full documentation, see
!   http://galahad.rl.ac.uk/galahad-www/specs.html

    MODULE GALAHAD_USEMIQR_double

!    -----------------------------------------------
!    | CUTEst/AMPL interface to MIQR, a multilevel |
!    | incomplete QR factorization package         |
!    -----------------------------------------------

!$    USE omp_lib
      USE CUTEst_interface_double
!NOT95USE GALAHAD_CPU_time
      USE GALAHAD_CLOCK
      USE GALAHAD_QPT_double
      USE GALAHAD_SORT_double, only: SORT_reorder_by_rows
      USE GALAHAD_NORMS_double, ONLY: TWO_NORM
      USE GALAHAD_MIQR_double
      USE GALAHAD_SPECFILE_double
      USE GALAHAD_STRING, ONLY: STRING_upper_word
      USE GALAHAD_COPYRIGHT
      USE GALAHAD_SYMBOLS,                                                     &
          ACTIVE                => GALAHAD_ACTIVE,                             &
          TRACE                 => GALAHAD_TRACE,                              &
          DEBUG                 => GALAHAD_DEBUG,                              &
          GENERAL               => GALAHAD_GENERAL,                            &
          ALL_ZEROS             => GALAHAD_ALL_ZEROS

      IMPLICIT NONE

      PRIVATE
      PUBLIC :: USE_MIQR

    CONTAINS

!-*-*-*-*-*-*-*-*-*-   U S E _ C Q P  S U B R O U T I N E   -*-*-*-*-*-*-*-*-

     SUBROUTINE USE_MIQR( input )

!  --------------------------------------------------------------------
!
!  Form a multilevel incomplete QR factorization of a rectangular
!  matrix A using the GALAHAD package GALAHAD_MIQR
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

!     INTEGER, PARAMETER :: n_k = 100, k_k = 3, in = 28
!     REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( :, : ) :: k_val
!     CHARACTER ( len = 10 ) :: filename = 'k.val'

!  Scalars

      INTEGER :: i, j, l, nea, n, m, la, liw, iores, smt_stat
      INTEGER :: status, alloc_stat, cutest_status, A_ne, iter, nm, maxc
      REAL :: time, timeo, times, timet
      REAL ( KIND = wp ) :: clock, clocko, clocks, clockt, objf, qfval, val
      LOGICAL :: filexx, printo, printe, is_specfile

!  Specfile characteristics

      INTEGER, PARAMETER :: input_specfile = 34
      INTEGER, PARAMETER :: lspec = 26
      CHARACTER ( LEN = 16 ) :: specname = 'RUNMIQR'
      TYPE ( SPECFILE_item_type ), DIMENSION( lspec ) :: spec
      CHARACTER ( LEN = 16 ) :: runspec = 'RUNMIQR.SPC'
      CHARACTER ( LEN = 4 ) :: rowcol

!  The default values for MIQR could have been set as:

! BEGIN RUNMIQR SPECIFICATIONS (DEFAULT)
!  write-problem-data                        NO
!  problem-data-file-name                    MIQR.data
!  problem-data-file-device                  26
!  write-initial-sif                         NO
!  initial-sif-file-name                     INITIAL.SIF
!  initial-sif-file-device                   51
!  least-squares-qp                          NO
!  solve-problem                             YES
!  print-full-solution                       NO
!  write-solution                            NO
!  solution-file-name                        MIQRSOL.d
!  solution-file-device                      62
!  write-result-summary                      NO
!  result-summary-file-name                  MIQRRES.d
!  result-summary-file-device                47
! END RUNMIQR SPECIFICATIONS

!  Default values for specfile-defined parameters

      INTEGER :: dfiledevice = 26
      INTEGER :: ifiledevice = 51
!     INTEGER :: pfiledevice = 50
!     INTEGER :: qfiledevice = 58
      INTEGER :: rfiledevice = 47
      INTEGER :: sfiledevice = 62
!     INTEGER :: lfiledevice = 78
      LOGICAL :: write_problem_data   = .FALSE.
      LOGICAL :: write_initial_sif    = .FALSE.
      LOGICAL :: write_solution       = .FALSE.
      LOGICAL :: write_result_summary = .FALSE.
      CHARACTER ( LEN = 30 ) :: dfilename = 'MIQR.data'
      CHARACTER ( LEN = 30 ) :: ifilename = 'INITIAL.SIF'
      CHARACTER ( LEN = 30 ) :: rfilename = 'MIQRRES.d'
      CHARACTER ( LEN = 30 ) :: sfilename = 'MIQRSOL.d'
!     CHARACTER ( LEN = 30 ) :: lfilename = 'LPROWS.d'
      LOGICAL :: do_solve = .TRUE.
      LOGICAL :: fulsol = .FALSE.

!  Output file characteristics

      INTEGER, PARAMETER :: out  = 6
      INTEGER, PARAMETER :: io_buffer = 11
      INTEGER :: errout = 6
      CHARACTER ( LEN =  5 ) :: solv
      CHARACTER ( LEN = 10 ) :: pname

!  Arrays

      TYPE ( MIQR_data_type ) :: data
      TYPE ( MIQR_control_type ) :: MIQR_control
      TYPE ( MIQR_inform_type ) :: MIQR_inform
      TYPE ( QPT_problem_type ) :: prob

!  Allocatable arrays

      CHARACTER ( LEN = 10 ), ALLOCATABLE, DIMENSION( : ) :: VNAME, CNAME
      REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: C
      LOGICAL, ALLOCATABLE, DIMENSION( : ) :: EQUATN, LINEAR
      INTEGER, ALLOCATABLE, DIMENSION( : ) :: IW, C_stat, B_stat, COUNT
      REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: RHS, SOL

      CALL CPU_TIME( time ) ; CALL CLOCK_time( clock )

!  ------------------ Open the specfile for miqr ----------------

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
        spec( 7 )%keyword = ''
        spec( 8 )%keyword = ''
        spec( 9 )%keyword = ''
        spec( 10 )%keyword = ''
        spec( 11 )%keyword = ''
        spec( 12 )%keyword = ''
        spec( 13 )%keyword = 'solve-problem'
        spec( 14 )%keyword = 'print-full-solution'
        spec( 15 )%keyword = 'write-solution'
        spec( 16 )%keyword = 'solution-file-name'
        spec( 17 )%keyword = 'solution-file-device'
        spec( 18 )%keyword = 'write-result-summary'
        spec( 19 )%keyword = 'result-summary-file-name'
        spec( 20 )%keyword = 'result-summary-file-device'
        spec( 21 )%keyword = ''
        spec( 22 )%keyword = ''
        spec( 23 )%keyword = ''
        spec( 24 )%keyword = ''
        spec( 25 )%keyword = ''
        spec( 26 )%keyword = ''

!   Read the specfile

        CALL SPECFILE_read( input_specfile, specname, spec, lspec, errout )

!   Interpret the result

        CALL SPECFILE_assign_logical( spec( 1 ), write_problem_data, errout )
        CALL SPECFILE_assign_string ( spec( 2 ), dfilename, errout )
        CALL SPECFILE_assign_integer( spec( 3 ), dfiledevice, errout )
        CALL SPECFILE_assign_logical( spec( 4 ), write_initial_sif, errout )
        CALL SPECFILE_assign_string ( spec( 5 ), ifilename, errout )
        CALL SPECFILE_assign_integer( spec( 6 ), ifiledevice, errout )
!       CALL SPECFILE_assign_integer( spec( 8 ), scale, errout )
!       CALL SPECFILE_assign_logical( spec( 9 ), do_presolve, errout )
!       CALL SPECFILE_assign_logical( spec( 10 ), write_presolved_sif, errout )
!       CALL SPECFILE_assign_string ( spec( 11 ), pfilename, errout )
!       CALL SPECFILE_assign_integer( spec( 12 ), pfiledevice, errout )
        CALL SPECFILE_assign_logical( spec( 13 ), do_solve, errout )
        CALL SPECFILE_assign_logical( spec( 14 ), fulsol, errout )
        CALL SPECFILE_assign_logical( spec( 15 ), write_solution, errout )
        CALL SPECFILE_assign_string ( spec( 16 ), sfilename, errout )
        CALL SPECFILE_assign_integer( spec( 17 ), sfiledevice, errout )
        CALL SPECFILE_assign_logical( spec( 18 ), write_result_summary, errout )
        CALL SPECFILE_assign_string ( spec( 19 ), rfilename, errout )
        CALL SPECFILE_assign_integer( spec( 20 ), rfiledevice, errout )
!       CALL SPECFILE_assign_logical( spec( 22 ), write_scaled_sif, errout )
!       CALL SPECFILE_assign_string ( spec( 23 ), qfilename, errout )
!       CALL SPECFILE_assign_integer( spec( 24 ), qfiledevice, errout )
!       CALL SPECFILE_assign_real( spec( 25 ), H_pert, errout )
!       CALL SPECFILE_assign_logical( spec( 26 ), convexify, errout )
      END IF

!  Determine the number of variables and constraints

      CALL CUTEST_cdimen( cutest_status, input, n, m )
      IF ( cutest_status /= 0 ) GO TO 910

!  Allocate suitable arrays

      ALLOCATE( prob%X( n ), prob%X_l( n ), prob%X_u( n ),                     &
                prob%G( n ), VNAME( n ), B_stat( n ), STAT = alloc_stat )
      IF ( alloc_stat /= 0 ) THEN
        WRITE( out, 2150 ) 'X', alloc_stat ; STOP
      END IF

      ALLOCATE( prob%C_l( m ), prob%C_u( m ), prob%Y( m ), CNAME( m ),         &
                EQUATN( m ), LINEAR( m ), C_stat( m ), STAT = alloc_stat )
      IF ( alloc_stat /= 0 ) THEN
        WRITE( out, 2150 ) 'C', alloc_stat ; STOP
      END IF

!  Set up the data structures necessary to hold the group partially
!  separable function.

      CALL CUTEST_csetup( cutest_status, input, out, io_buffer,                &
                          n, m, prob%X, prob%X_l, prob%X_u,                    &
                          prob%Y, prob%C_l, prob%C_u, EQUATN, LINEAR, 0, 0, 0 )
      IF ( cutest_status /= 0 ) GO TO 910
      DEALLOCATE( LINEAR )

!  Allocate derived types

      ALLOCATE( prob%X0( n ), STAT = alloc_stat )
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

      prob%X0 = zero

!  Evaluate the constant terms of the objective (objf) and constraint
!  functions (C)

      CALL CUTEST_cfn( cutest_status, n, m, prob%X0, objf, C( : m ) )
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

!  Evaluate the problem gradients

      CALL CUTEST_csgr( cutest_status, n, m, prob%X0, prob%Y, .FALSE.,         &
                        nea, la, prob%A%val, prob%A%col, prob%A%row )
      IF ( cutest_status /= 0 ) GO TO 910

!  Exclude zeros; extract the linear term for the objective function and the
!  constraint Jacobian

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
      prob%A%n = n ; prob%A%m = m ; prob%A%ne = A_ne

!  Allocate and initialize dual variables.

      ALLOCATE( prob%Z( n ), STAT = alloc_stat )
      IF ( alloc_stat /= 0 ) THEN
        WRITE( out, 2150 ) 'Z', alloc_stat
        STOP
      END IF
      prob%Z( : n ) = one

      liw = MAX( m, n ) + 1
      ALLOCATE( prob%A%ptr( m + 1 ) )
      ALLOCATE( IW( liw ) )

!  Transform A to row storage format

      IF ( A_ne /= 0 ) THEN
        CALL SORT_reorder_by_rows( m, n, A_ne, prob%A%row, prob%A%col, A_ne,   &
                                   prob%A%val, prob%A%ptr, m + 1, IW, liw,     &
                                   out, out, i )
      ELSE
        prob%A%ptr = 0
      END IF

!  Deallocate arrays holding matrix row indices

      DEALLOCATE( prob%A%row )
      ALLOCATE( prob%A%row( 0 ) )

      prob%new_problem_structure = .TRUE.

!  Store the problem dimensions

      prob%n = n ; prob%m = m
      IF ( ALLOCATED( prob%A%type ) ) DEALLOCATE( prob%A%type )
      CALL SMT_put( prob%A%type, 'SPARSE_BY_ROWS', smt_stat )
      prob%f    = objf

!  ------------------- problem set-up complete ----------------------

      CALL CPU_TIME( times ) ; CALL CLOCK_time( clocks )

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

      CALL MIQR_initialize( data, MIQR_control, MIQR_inform )

      IF ( is_specfile )                                                       &
        CALL MIQR_read_specfile( MIQR_control, input_specfile )

      printo = out > 0 .AND. MIQR_control%print_level > 0
      printe = out > 0 .AND. MIQR_control%print_level >= 0
      WRITE( out, 2200 ) n, m, A_ne

      IF ( printo ) CALL COPYRIGHT( out, '2014' )

!  compute the number of nonzeros in each column

       IF ( MIQR_control%transpose ) THEN
         IW( : n ) = 0
         DO i = 1, m
           DO l = prob%A%ptr( i ), prob%A%ptr( i + 1 ) - 1
             j = prob%A%col( l )
             IW( j ) = IW( j ) + 1
           END DO
         END DO
         maxc = MAXVAL( IW( : n ) )
         nm = n
         rowcol = 'cols'

!  compute the number of nonzeros in each row

      ELSE
        maxc = 0
        DO i = 1, m
          IW( i ) = prob%A%ptr( i + 1 ) - prob%A%ptr( i )
          maxc = MAX( maxc, IW( i ) )
        END DO
        nm = m
        rowcol = 'rows'
      END IF

      ALLOCATE( COUNT( 0 : maxc ) )
      COUNT = 0
      DO i = 1, nm
        j = IW( i )
        COUNT( j ) = COUNT( j ) + 1
      END DO

!     INQUIRE( FILE = lfilename, EXIST = filexx )
!     IF ( filexx ) THEN
!        OPEN( lfiledevice, FILE = lfilename, FORM = 'FORMATTED',            &
!              STATUS = 'OLD', POSITION = 'APPEND', IOSTAT = iores )
!     ELSE
!        OPEN( lfiledevice, FILE = lfilename, FORM = 'FORMATTED',            &
!              STATUS = 'NEW', IOSTAT = iores )
!     END IF
!     IF ( iores /= 0 ) THEN
!       write( out, 2160 ) iores, lfilename
!       STOP
!     END IF
!     WRITE( lfiledevice, "( ' ------------------------------------' )" )
!     WRITE( lfiledevice, "( 1X, A10, ' m = ', I8, ' n = ', I8 )" ) pname, m, n

      DO j = 0, maxc
        IF ( COUNT( j ) > 0 ) THEN
          WRITE( out, "( 1X, I0, 1X, A, ' have ', I0, ' nonzeros' )" )         &
            COUNT( j ), rowcol, j
!         WRITE( lfiledevice, "( 1X, I0, 1X, A, ' have ', I0, ' nonzeros' )" ) &
!           COUNT( j ), rowcol, j
        END IF
      END DO
      DEALLOCATE( IW, COUNT )
!     CLOSE( lfiledevice )
!     STOP

      C_stat = 0 ; B_stat = 0 ; prob%C = zero

!  If the problem to be output, allocate sufficient space

      IF ( write_initial_sif ) THEN

        ALLOCATE( prob%X_status( n ), STAT = alloc_stat )
        IF ( alloc_stat /= 0 ) THEN
          IF ( printe ) WRITE( out, 2150 ) 'X_status', alloc_stat
          STOP
        END IF
        prob%X_status = ACTIVE

        ALLOCATE( prob%C_status( m ), STAT = alloc_stat )
        IF ( alloc_stat /= 0 ) THEN
          IF ( printe ) WRITE( out, 2150 ) 'X_status', alloc_stat
          STOP
        END IF
        prob%C_status = ACTIVE

        ALLOCATE( prob%Z_l( n ), prob%Z_u( n ), STAT = alloc_stat )
        IF ( alloc_stat /= 0 ) THEN
          IF ( printe ) WRITE( out, 2150 ) 'Z_lu', alloc_stat
          STOP
        END IF
        prob%Z_l( : n ) = - infinity
        prob%Z_u( : n ) =   infinity

        ALLOCATE( prob%Y_l( m ), prob%Y_u( m ), STAT = alloc_stat )
        IF ( alloc_stat /= 0 ) THEN
          IF ( printe ) WRITE( out, 2150 ) 'C_lu', alloc_stat
          STOP
        END IF
        prob%Y_l( : m ) = - infinity
        prob%Y_u( : m ) =   infinity

!  Writes the initial SIF file, if needed

        IF ( write_initial_sif ) THEN
          CALL QPT_write_to_sif( prob, pname, ifilename, ifiledevice,          &
                                 .FALSE., .FALSE., infinity )
          IF ( .NOT. do_solve ) STOP
        END IF
      END IF

!  Call the factorization package

      qfval = objf

      IF ( do_solve .AND. prob%n > 0 ) THEN
        CALL CPU_TIME( timeo ) ; CALL CLOCK_time( clocko )

        prob%m = m ; prob%n = n
        DEALLOCATE( prob%X0 )

!       prob%m = m
!       prob%n = n

!       WRITE( 6, "( ' x ', /, (5ES12.4) )" ) prob%X
!       WRITE( 6, "( ' y ', /, (5ES12.4) )" ) prob%Y
!       WRITE( 6, "( ' z ', /, (5ES12.4) )" ) prob%Z

!  =================
!  solve the problem
!  =================

        solv = ' MIQR'
        IF ( printo ) WRITE( out, " ( ' ** MIQR solver used ** ' ) " )

!  factorize the matrix

        CALL MIQR_form( prob%A, data, MIQR_control, MIQR_inform )
        WRITE( 6, "( ' on exit from MIQR_form, status = ', I0,                 &
       &    ', dropped = ', I0, ', time = ', F6.2 )" )                         &
          MIQR_inform%status, MIQR_inform%drop, MIQR_inform%time%clock_form
        IF ( MIQR_inform%zero_diagonals > 0 )                                  &
          WRITE( 6, "( 1X, I0, ' zero columns ' )" ) MIQR_inform%zero_diagonals

        IF ( MIQR_control%transpose ) THEN
          IF ( printo ) WRITE( out,                                            &
            " ( /, ' ** MIQR solver used on transpose ** ' ) " )
        ELSE
          IF ( printo ) WRITE( out, " ( /, ' ** MIQR solver used ** ' ) " )
        END IF

        ALLOCATE( RHS( MAX( m, n ) ), SOL( MAX( m, n ) ), STAT = alloc_stat )

!  form the RHS = A A^T e

        IF ( MIQR_control%transpose ) THEN
          RHS( : m ) = one
          SOL( : n ) = zero
          DO i = 1, m
            val = RHS( i )
            DO l = prob%A%ptr( i ), prob%A%ptr( i + 1 ) - 1
              j = prob%A%col( l )
              SOL( j ) = SOL( j ) + prob%A%val( l ) * val
            END DO
          END DO
          DO i = 1, m
            val = zero
            DO l = prob%A%ptr( i ), prob%A%ptr( i + 1 ) - 1
              val = val + prob%A%val( l ) * SOL( prob%A%col( l ) )
            END DO
            RHS( i ) = val
          END DO
          i = n ; n = m ; m = i

!  form the RHS = A^T A e

        ELSE
          RHS( : n ) = one
          DO i = 1, m
            val = zero
            DO l = prob%A%ptr( i ), prob%A%ptr( i + 1 ) - 1
              val = val + prob%A%val( l ) * RHS( prob%A%col( l ) )
            END DO
            SOL( i ) = val
          END DO
          RHS( : n ) = zero
          DO i = 1, m
            val = SOL( i )
            DO l = prob%A%ptr( i ), prob%A%ptr( i + 1 ) - 1
              j = prob%A%col( l )
              RHS( j ) = RHS( j ) + prob%A%val( l ) * val
            END DO
          END DO
        END IF

        WRITE( 6, "( ' ||rhs||            =', ES11.4 )" )                      &
          MAXVAL( ABS ( RHS( : n ) ) )

! solve the system R^T sol = rhs

        CALL MIQR_apply( RHS, .TRUE., data, MIQR_inform )
        WRITE( 6, "( ' ||sol(transpose)|| =', ES11.4 )" )                      &
          MAXVAL( ABS ( RHS( : n ) ) )
!       WRITE( 6, "( ' sol(transpose) =', /, ( 5ES11.4 ) )" ) RHS( : n )

! solve the system R rhs = sol

        CALL MIQR_apply( RHS, .FALSE., data, MIQR_inform )
        WRITE( 6, "( ' ||sol||            =', ES11.4 )" )                      &
          MAXVAL( ABS ( RHS( : n ) ) )
!       WRITE( 6, "( ' sol =', /, ( 5ES11.4 ) )" ) RHS( : n )

        CALL CPU_TIME( timet ) ; CALL CLOCK_time( clockt )

!  Deallocate arrays from the minimization

        status = MIQR_inform%status
        CALL MIQR_terminate( data, MIQR_control, MIQR_inform )
        DEALLOCATE( RHS, SOL )

      ELSE
        timeo  = 0.0
        timet  = 0.0
        iter  = 0
        solv   = ' NONE'
        status = 0
        qfval  = prob%f
      END IF

      DEALLOCATE( VNAME, CNAME, C )
      IF ( is_specfile ) CLOSE( input_specfile )

      CALL CUTEST_cterminate( cutest_status )
      RETURN

 910  CONTINUE
      WRITE( out, "( ' CUTEst error, status = ', i0, ', stopping' )" )         &
        cutest_status
      status = - 98
      RETURN

!  Non-executable statements

 2020 FORMAT( /, ' Problem: ', A )
 2150 FORMAT( ' Allocation error, variable ', A8, ' status = ', I0 )
 2160 FORMAT( ' IOSTAT = ', I6, ' when opening file ', A9, '. Stopping ' )
 2180 FORMAT( A10 )
 2200 FORMAT( /, ' problem dimensions:  n = ', I0, ', m = ', I0,               &
              ', a_ne = ', I0 )

!  End of subroutine USE_MIQR

     END SUBROUTINE USE_MIQR

!  End of module USEMIQR_double

   END MODULE GALAHAD_USEMIQR_double
