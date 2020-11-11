! THIS VERSION: GALAHAD 3.3 - 29/10/2020 AT 08:30 GMT.

!-*-*-*-*-*-*-*-  G A L A H A D   U S E L S T R   M O D U L E  -*-*-*-*-*-*-*-*-

!  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
!  Principal authors: Nick Gould and Dominique Orban

!  History -
!   originally released with GALAHAD Version 2.6. May 27th 2014

!  For full documentation, see
!   http://galahad.rl.ac.uk/galahad-www/specs.html

    MODULE GALAHAD_USELSTR_double

!     ----------------------------------------------------
!    | CUTEst/AMPL interface to LSTR, an iterative method |
!    | for trust-region regularized linear least squares  |
!     ----------------------------------------------------

!$    USE omp_lib
      USE CUTEst_interface_double
!NOT95USE GALAHAD_CPU_time
      USE GALAHAD_CLOCK
      USE GALAHAD_SMT_double
      USE GALAHAD_QPT_double
      USE GALAHAD_SORT_double, only: SORT_reorder_by_rows
      USE GALAHAD_NORMS_double, ONLY: TWO_NORM
      USE GALAHAD_LSTR_double
      USE GALAHAD_SPECFILE_double
      USE GALAHAD_STRING, ONLY: STRING_upper_word
      USE GALAHAD_COPYRIGHT
      USE GALAHAD_CONVERT_double
      USE GALAHAD_SYMBOLS,                                                     &
          ACTIVE                => GALAHAD_ACTIVE,                             &
          TRACE                 => GALAHAD_TRACE,                              &
          DEBUG                 => GALAHAD_DEBUG,                              &
          GENERAL               => GALAHAD_GENERAL,                            &
          ALL_ZEROS             => GALAHAD_ALL_ZEROS
      USE GALAHAD_MIQR_double
      USE hsl_mi35_double

      IMPLICIT NONE

      PRIVATE
      PUBLIC :: USE_LSTR

    CONTAINS

!-*-*-*-*-*-*-*-*-*-   U S E _ C Q P  S U B R O U T I N E   -*-*-*-*-*-*-*-*-

     SUBROUTINE USE_LSTR( input )

!  --------------------------------------------------------------------
!
!  Form a multilevel incomplete QR factorization of a rectangular
!  matrix A using the GALAHAD package GALAHAD_LSTR
!
!  --------------------------------------------------------------------

!  Dummy argument

      INTEGER, INTENT( IN ) :: input

!  Parameters

      INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
      INTEGER, PARAMETER :: long = SELECTED_INT_KIND( 18 )
      REAL ( KIND = wp ), PARAMETER :: zero = 0.0_wp
      REAL ( KIND = wp ), PARAMETER :: one = 1.0_wp
      REAL ( KIND = wp ), PARAMETER :: ten = 10.0_wp
      REAL ( KIND = wp ), PARAMETER :: infinity = ten ** 19
      REAL ( KIND = wp ), PARAMETER :: d_min = 0.0001_wp

      INTEGER, PARAMETER :: no_preconditioner = 0
      INTEGER, PARAMETER :: diagonal_preconditioner = 1
      INTEGER, PARAMETER :: miqr_preconditioner = 2
      INTEGER, PARAMETER :: mi35_preconditioner = 7
      INTEGER, PARAMETER :: mi35_with_c_preconditioner = 8

!     INTEGER, PARAMETER :: n_k = 100, k_k = 3, in = 28
!     REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( :, : ) :: k_val
!     CHARACTER ( len = 10 ) :: filename = 'k.val'

!  Scalars

      INTEGER :: i, j, l, nea, n, m, la, liw, iores, smt_stat, m_used, n_used
      INTEGER :: status, alloc_stat, cutest_status, A_ne, maxc, ns, n_total
      INTEGER :: size_r, size_l, branch
      INTEGER ( KIND = long ) :: n_fact = - 1
      REAL :: time, timep, times, timet
      REAL ( KIND = wp ) :: clock, clockp, clocks, clockt
      REAL ( KIND = wp ) :: objf, val, x_norm
      LOGICAL :: filexx, printo, printe, is_specfile

!  Specfile characteristics

      INTEGER, PARAMETER :: input_specfile = 34
      INTEGER, PARAMETER :: lspec = 37
      CHARACTER ( LEN = 16 ) :: specname = 'RUNLSTR'
      TYPE ( SPECFILE_item_type ), DIMENSION( lspec ) :: spec
      CHARACTER ( LEN = 16 ) :: runspec = 'RUNLSTR.SPC'
      CHARACTER ( LEN = 4 ) :: rowcol

!  The default values for LSTR could have been set as:

! BEGIN RUNLSTR SPECIFICATIONS (DEFAULT)
!  write-problem-data                            NO
!  problem-data-file-name                        LSTR.data
!  problem-data-file-device                      26
!  write-initial-sif                             NO
!  initial-sif-file-name                         INITIAL.SIF
!  initial-sif-file-device                       51
!  solve-problem                                 YES
!  solve-consistent-system                       NO
!  transpose-problem                             NO
!  preconditioner                                0
!  mi35-size-l                                   750
!  mi35-size-r                                   750
!  mi35-scale                                    1
!  mi35-tau1                                     0.0
!  mi35-tau2                                     0.0
!  trust-region-radius                           1.0D+20
!  print-full-solution                           NO
!  write-solution                                NO
!  solution-file-name                            LSTRSOL.d
!  solution-file-device                          62
!  write-result-summary                          NO
!  result-summary-file-name                      LSTRRES.d
!  result-summary-file-device                    47
! END RUNLSTR SPECIFICATIONS

!  Default values for specfile-defined parameters

      INTEGER :: dfiledevice = 26
      INTEGER :: ifiledevice = 51
!     INTEGER :: pfiledevice = 50
!     INTEGER :: qfiledevice = 58
      INTEGER :: rfiledevice = 47
      INTEGER :: sfiledevice = 62
!     INTEGER :: lfiledevice = 78
      LOGICAL :: consistent = .FALSE.
      LOGICAL :: write_problem_data   = .FALSE.
      LOGICAL :: write_initial_sif    = .FALSE.
      LOGICAL :: write_solution       = .FALSE.
      LOGICAL :: write_result_summary = .FALSE.
      CHARACTER ( LEN = 30 ) :: dfilename = 'LSTR.data'
      CHARACTER ( LEN = 30 ) :: ifilename = 'INITIAL.SIF'
      CHARACTER ( LEN = 30 ) :: rfilename = 'LSTRRES.d'
      CHARACTER ( LEN = 30 ) :: sfilename = 'LSTRSOL.d'
!     CHARACTER ( LEN = 30 ) :: lfilename = 'LPROWS.d'
      LOGICAL :: do_solve = .TRUE.
      LOGICAL :: fulsol = .FALSE.
      LOGICAL :: transpose = .FALSE.
      INTEGER :: preconditioner = 0
      INTEGER :: mi35_size_l = 20
      INTEGER :: mi35_size_r = 20
      INTEGER :: mi35_scale = 1
      REAL ( KIND = wp ) :: radius = ten ** 20
      REAL ( KIND = wp ) :: mi35_tau1 = 0.001_wp
      REAL ( KIND = wp ) :: mi35_tau2 = 0.001_wp

!  Output file characteristics

      INTEGER, PARAMETER :: out  = 6
      INTEGER, PARAMETER :: io_buffer = 11
      INTEGER :: errout = 6
!     CHARACTER ( LEN =  5 ) :: solv
      CHARACTER ( LEN = 10 ) :: pname

!  Arrays

      TYPE ( SMT_type ) :: A_by_cols, S, A_ls
      TYPE ( LSTR_data_type ) :: LSTR_data
      TYPE ( LSTR_control_type ) :: LSTR_control
      TYPE ( LSTR_inform_type ) :: LSTR_inform
      TYPE ( MIQR_data_type ) :: MIQR_data
      TYPE ( MIQR_control_type ) :: MIQR_control
      TYPE ( MIQR_inform_type ) :: MIQR_inform
      TYPE ( QPT_problem_type ) :: prob
      TYPE ( CONVERT_control_type ) :: CONVERT_control
      TYPE ( CONVERT_inform_type ) :: CONVERT_inform
      TYPE ( mi35_keep ) :: keep_mi35
      TYPE ( mi35_control ) :: control_mi35
      TYPE ( mi35_info ) :: info_mi35

!  Allocatable arrays

!     CHARACTER ( LEN = 10 ), ALLOCATABLE, DIMENSION( : ) :: VNAME, CNAME
      LOGICAL, ALLOCATABLE, DIMENSION( : ) :: EQUATN, LINEAR
      INTEGER, ALLOCATABLE, DIMENSION( : ) :: IW, NNZ_COUNT
      REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: X, V, W, D, RHS
      REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: B, U, RES

      CALL CPU_TIME( time ) ; CALL CLOCK_time( clock )

!  ------------------ Open the specfile for LSTR ----------------

      INQUIRE( FILE = runspec, EXIST = is_specfile )
      IF ( is_specfile ) THEN
        OPEN( input_specfile, FILE = runspec, FORM = 'FORMATTED',              &
              STATUS = 'OLD' )

!   Define the keywords

        spec(  1 )%keyword = 'write-problem-data'
        spec(  2 )%keyword = 'problem-data-file-name'
        spec(  3 )%keyword = 'problem-data-file-device'
        spec(  4 )%keyword = 'write-initial-sif'
        spec(  5 )%keyword = 'initial-sif-file-name'
        spec(  6 )%keyword = 'initial-sif-file-device'
        spec(  7 )%keyword = 'solve-consistent-system'
        spec(  8 )%keyword = 'preconditioner'
        spec(  9 )%keyword = 'mi35-size-l'
        spec( 10 )%keyword = 'mi35-size-r'
        spec( 11 )%keyword = 'mi35-scale'
        spec( 12 )%keyword = 'solve-problem'
        spec( 13 )%keyword = 'transpose-problem'
        spec( 14 )%keyword = 'print-full-solution'
        spec( 15 )%keyword = 'write-solution'
        spec( 16 )%keyword = 'solution-file-name'
        spec( 17 )%keyword = 'solution-file-device'
        spec( 18 )%keyword = 'write-result-summary'
        spec( 19 )%keyword = 'result-summary-file-name'
        spec( 20 )%keyword = 'result-summary-file-device'
        spec( 21 )%keyword = 'mi35-tau1'
        spec( 22 )%keyword = 'mi35-tau2'
        spec( 23 )%keyword = 'trust-region-radius'
        spec( 24 )%keyword = ''
        spec( 25 )%keyword = ''
        spec( 26 )%keyword = ''
        spec( 27 )%keyword = ''
        spec( 28 )%keyword = ''
        spec( 29 )%keyword = ''
        spec( 30 )%keyword = ''
        spec( 31 )%keyword = ''
        spec( 32 )%keyword = ''
        spec( 33 )%keyword = ''
        spec( 34 )%keyword = ''
        spec( 35 )%keyword = ''
        spec( 36 )%keyword = ''
        spec( 37 )%keyword = ''

!   Read the specfile

        CALL SPECFILE_read( input_specfile, specname, spec, lspec, errout )

!   Interpret the result

        CALL SPECFILE_assign_logical( spec( 1 ), write_problem_data, errout )
        CALL SPECFILE_assign_string ( spec( 2 ), dfilename, errout )
        CALL SPECFILE_assign_integer( spec( 3 ), dfiledevice, errout )
        CALL SPECFILE_assign_logical( spec( 4 ), write_initial_sif, errout )
        CALL SPECFILE_assign_string ( spec( 5 ), ifilename, errout )
        CALL SPECFILE_assign_integer( spec( 6 ), ifiledevice, errout )
        CALL SPECFILE_assign_logical( spec( 7 ), consistent, errout )
        CALL SPECFILE_assign_integer( spec( 8 ), preconditioner, errout )
        CALL SPECFILE_assign_integer( spec( 9 ), mi35_size_l, errout )
        CALL SPECFILE_assign_integer( spec( 10 ), mi35_size_r, errout )
        CALL SPECFILE_assign_integer( spec( 11 ), mi35_scale, errout )
        CALL SPECFILE_assign_logical( spec( 12 ), do_solve, errout )
        CALL SPECFILE_assign_logical( spec( 13 ), transpose, errout )
        CALL SPECFILE_assign_logical( spec( 14 ), fulsol, errout )
        CALL SPECFILE_assign_logical( spec( 15 ), write_solution, errout )
        CALL SPECFILE_assign_string ( spec( 16 ), sfilename, errout )
        CALL SPECFILE_assign_integer( spec( 17 ), sfiledevice, errout )
        CALL SPECFILE_assign_logical( spec( 18 ), write_result_summary, errout )
        CALL SPECFILE_assign_string ( spec( 19 ), rfilename, errout )
        CALL SPECFILE_assign_integer( spec( 20 ), rfiledevice, errout )
        CALL SPECFILE_assign_real( spec( 21 ), mi35_tau1, errout )
        CALL SPECFILE_assign_real( spec( 22 ), mi35_tau2, errout )
        CALL SPECFILE_assign_real( spec( 23 ), radius, errout )
      END IF

!  Determine the number of variables and constraints

      CALL CUTEST_cdimen( cutest_status, input, n, m )
      IF ( cutest_status /= 0 ) GO TO 910

!  Allocate suitable arrays

      ALLOCATE( prob%X( n ), prob%X_l( n ), prob%X_u( n ), prob%G( n ),        &
!               VNAME( n ),                                                    &
                STAT = alloc_stat )
      IF ( alloc_stat /= 0 ) THEN
        WRITE( out, 2150 ) 'X', alloc_stat ; STOP
      END IF

      ALLOCATE( prob%C_l( m ), prob%C_u( m ), prob%Y( m ),                     &
!               CNAME( m ),                                                    &
                EQUATN( m ), LINEAR( m ), STAT = alloc_stat )
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

      ALLOCATE( prob%C( m ), STAT = alloc_stat )
      IF ( alloc_stat /= 0 ) THEN
        WRITE( out, 2150 ) 'C', alloc_stat
        STOP
      END IF

!  Determine the names of the problem, variables and constraints.

      CALL CUTEST_probname( cutest_status, pname )
!     CALL CUTEST_cnames( cutest_status, n, m, pname, VNAME, CNAME )
      IF ( cutest_status /= 0 ) GO TO 910
      WRITE( out, "( ' Problem: ', A )" ) pname

!  Set up the initial estimate of the solution and
!  right-hand-side of the Kuhn-Tucker system.

!  Determine the constant terms for the problem functions.

      prob%X( : n ) = MIN( prob%X_u( : n ),                                    &
                           MAX( prob%X_l( : n ), prob%X( : n ) ) )

!  Set X0 to zero to determine the constant terms for the problem functions

      prob%X0 = zero

!  Evaluate the constant terms of the objective (objf) and constraint
!  functions (C)

      CALL CUTEST_cfn( cutest_status, n, m, prob%X0, objf, prob%C( : m ) )
      IF ( cutest_status /= 0 ) GO TO 910
!     DO i = 1, m
!       IF ( EQUATN( i ) ) THEN
!         prob%C_l( i ) = prob%C_l( i ) - prob%C( i )
!         prob%C_u( i ) = prob%C_l( i )
!       ELSE
!         prob%C_l( i ) = prob%C_l( i ) - prob%C( i )
!         prob%C_u( i ) = prob%C_u( i ) - prob%C( i )
!       END IF
!     END DO

!  Determine the number of nonzeros in the Jacobian

      CALL CUTEST_cdimsj( cutest_status, la )
      ns = COUNT( .NOT. EQUATN( : m ) )
      IF ( cutest_status /= 0 ) GO TO 910
      la = MAX( la + ns, 1 )

!  Allocate arrays to hold the Jacobian

      ALLOCATE( prob%A%row( la ), prob%A%col( la ), prob%A%val( la ),          &
                STAT = alloc_stat )
      IF ( alloc_stat /= 0 ) THEN
        WRITE( out, 2150 ) 'A', alloc_stat ; STOP
      END IF

!  introduce slack variables for inequalities

!     A_ne = 0
!     n_total = 0
!     DO i = 1, m
!       IF ( .NOT. EQUATN( i ) ) THEN
!         n_total = n_total + 1
!         A_ne = A_ne + 1
!         prob%A%row( A_ne ) = i
!         prob%A%col( A_ne ) = n_total
!         prob%A%val( A_ne ) = - one
!       END IF
!     END DO

!  evaluate the problem gradients

!     CALL CUTEST_csgr( cutest_status, n, m, prob%X0, prob%Y, .FALSE.,         &
!                       nea, la - ns , prob%A%val( ns + 1 : ),                 &
!                       prob%A%col( ns + 1 : ), prob%A%row( ns + 1 : ) )
!     IF ( cutest_status /= 0 ) GO TO 910

      CALL CUTEST_csgr( cutest_status, n, m, prob%X0, prob%Y, .FALSE.,         &
                        nea, la - ns , prob%A%val( : la - ns ),                &
                        prob%A%col( : la - ns ), prob%A%row( : la - ns ) )
      IF ( cutest_status /= 0 ) GO TO 910

!  exclude zeros; extract the linear term for the objective function and the
!  constraint Jacobian

      prob%G( : n ) = zero
      A_ne = 0
!     DO i = ns + 1, ns + nea
      DO i = 1, nea
        IF ( prob%A%val( i ) /= zero ) THEN
          IF ( prob%A%row( i ) > 0 ) THEN
            A_ne = A_ne + 1
            prob%A%row( A_ne ) = prob%A%row( i )
            prob%A%col( A_ne ) = prob%A%col( i ) + n_total
            prob%A%val( A_ne ) = prob%A%val( i )
          ELSE
            prob%G( prob%A%col( i ) ) = prob%A%val( i )
          END IF
        END IF
      END DO
      n_total = n_total + n

!  introduce slack variables for inequalities

      n_total = n
      DO i = 1, m
        IF ( .NOT. EQUATN( i ) ) THEN
          n_total = n_total + 1
          A_ne = A_ne + 1
          prob%A%row( A_ne ) = i
          prob%A%col( A_ne ) = n_total
          prob%A%val( A_ne ) = - one
        END IF
      END DO

      prob%A%n = n_total ; prob%A%m = m ; prob%A%ne = A_ne

!OPEN(99)
!write(99,"('m,n,ne')")
!write(99,*) prob%A%m, prob%A%n, prob%A%ne
!write(99,"('row,col,val')")
!DO i = 1, prob%A%ne
!  write(99,*) prob%A%row( i ), prob%A%col( i ), prob%A%val( i )
!END DO
!CLOSE(99)

      liw = MAX( m, n_total ) + 1
!     DEALLOCATE( VNAME, CNAME )
      ALLOCATE( prob%A%ptr( m + 1 ) )
      ALLOCATE( IW( liw ) )

!  Transform A to row storage format

      IF ( A_ne /= 0 ) THEN
        CALL SORT_reorder_by_rows( m, n_total, A_ne, prob%A%row, prob%A%col,   &
                                   A_ne, prob%A%val, prob%A%ptr, m + 1,        &
                                   IW, liw, out, out, i )
      ELSE
        prob%A%ptr = 0
      END IF

!  Deallocate arrays holding matrix row indices

!     DEALLOCATE( prob%A%row )
!     ALLOCATE( prob%A%row( 0 ) )

      prob%new_problem_structure = .TRUE.

!  Store the problem dimensions

      prob%n = n ; prob%m = m
      IF ( ALLOCATED( prob%A%type ) ) DEALLOCATE( prob%A%type )
      CALL SMT_put( prob%A%type, 'SPARSE_BY_ROWS', smt_stat )
      prob%f = objf

!  ------------------- problem set-up complete ----------------------

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

!  compute the number of nonzeros in each column

       IF ( transpose ) THEN
         IW( : n_total ) = 0
         DO i = 1, m
           DO l = prob%A%ptr( i ), prob%A%ptr( i + 1 ) - 1
             j = prob%A%col( l )
             IW( j ) = IW( j ) + 1
           END DO
         END DO
         maxc = MAXVAL( IW( : n_total ) )
         m_used = n_total ; n_used = m
         rowcol = 'cols'

!  compute the number of nonzeros in each row

      ELSE
        maxc = 0
        DO i = 1, m
          IW( i ) = prob%A%ptr( i + 1 ) - prob%A%ptr( i )
          maxc = MAX( maxc, IW( i ) )
        END DO
        m_used = m ; n_used = n_total
        rowcol = 'rows'
      END IF

      ALLOCATE( NNZ_COUNT( 0 : maxc ) )
      NNZ_COUNT = 0
      DO i = 1, m_used
        j = IW( i )
        NNZ_COUNT( j ) = NNZ_COUNT( j ) + 1
      END DO

!     INQUIRE( FILE = lfilename, EXIST = filexx )
!     IF ( filexx ) THEN
!        OPEN( lfiledevice, FILE = lfilename, FORM = 'FORMATTED',              &
!              STATUS = 'OLD', POSITION = 'APPEND', IOSTAT = iores )
!     ELSE
!        OPEN( lfiledevice, FILE = lfilename, FORM = 'FORMATTED',              &
!              STATUS = 'NEW', IOSTAT = iores )
!     END IF
!     IF ( iores /= 0 ) THEN
!       write( out, 2160 ) iores, lfilename
!       STOP
!     END IF
!     WRITE( lfiledevice, "( ' ------------------------------------' )" )
!     WRITE( lfiledevice, "( 1X, A10, ' m = ', I8, ' n = ', I8 )" ) pname, m, n

      DO j = 0, maxc
        IF ( NNZ_COUNT( j ) > 0 ) THEN
!         WRITE( out, "( 1X, I0, 1X, A, ' have ', I0, ' nonzeros' )" )         &
!           NNZ_COUNT( j ), rowcol, j
!         WRITE( lfiledevice, "( 1X, I0, 1X, A, ' have ', I0, ' nonzeros' )" ) &
!           NNZ_COUNT( j ), rowcol, j
        END IF
      END DO
      DEALLOCATE( NNZ_COUNT )
!     CLOSE( lfiledevice )
!     STOP

!  Set all default values, and override defaults if requested

      CALL LSTR_initialize( LSTR_data, LSTR_control, LSTR_inform )

      IF ( is_specfile )                                                       &
        CALL LSTR_read_specfile( LSTR_control, input_specfile )

      printo = out > 0 .AND. LSTR_control%print_level > 0
      printe = out > 0 .AND. LSTR_control%print_level >= 0
      WRITE( out, 2200 ) n, ns, m, A_ne

      IF ( printo ) CALL COPYRIGHT( out, '2014' )

!  If the problem to be output, allocate sufficient space

!     IF ( write_initial_sif ) THEN

!       ALLOCATE( prob%X_status( n ), STAT = alloc_stat )
!        IF ( alloc_stat /= 0 ) THEN
!         IF ( printe ) WRITE( out, 2150 ) 'X_status', alloc_stat
!         STOP
!       END IF
!       prob%X_status = ACTIVE

!       ALLOCATE( prob%C_status( m ), STAT = alloc_stat )
!       IF ( alloc_stat /= 0 ) THEN
!         IF ( printe ) WRITE( out, 2150 ) 'X_status', alloc_stat
!         STOP
!       END IF
!       prob%C_status = ACTIVE

!       ALLOCATE( prob%Z_l( n ), prob%Z_u( n ), STAT = alloc_stat )
!       IF ( alloc_stat /= 0 ) THEN
!         IF ( printe ) WRITE( out, 2150 ) 'Z_lu', alloc_stat
!         STOP
!        END IF
!       prob%Z_l( : n ) = - infinity
!       prob%Z_u( : n ) =   infinity

!       ALLOCATE( prob%Y_l( m ), prob%Y_u( m ), STAT = alloc_stat )
!       IF ( alloc_stat /= 0 ) THEN
!         IF ( printe ) WRITE( out, 2150 ) 'C_lu', alloc_stat
!         STOP
!       END IF
!       prob%Y_l( : m ) = - infinity
!       prob%Y_u( : m ) =   infinity

!  Writes the initial SIF file, if needed

!       IF ( write_initial_sif ) THEN
!         CALL QPT_write_to_sif( prob, pname, ifilename, ifiledevice,          &
!                                .FALSE., .FALSE., infinity )
!         IF ( .NOT. do_solve ) STOP
!       END IF
!     END IF

      ALLOCATE( X( n_used ), V( n_used ), W( MAX( n_used, m_used ) ),          &
                B( m_used ), U( m_used ), RES( m_used ), STAT = alloc_stat )
      IF ( alloc_stat /= 0 ) THEN
        IF ( printe ) WRITE( out, 2150 ) 'X, V, U, RES', alloc_stat
        STOP
      END IF

!  set the right-hand side so that the consistent solution is one

      IF ( consistent ) THEN
        IF ( transpose ) THEN
          B( : n_used ) = zero
          DO i = 1, m
            DO l = prob%A%ptr( i ), prob%A%ptr( i + 1 ) - 1
              j = prob%A%col( l )
              B( j ) = B( j ) + prob%A%val( l ) * one
            END DO
          END DO
        ELSE
          DO i = 1, m
            val = zero
            DO l = prob%A%ptr( i ), prob%A%ptr( i + 1 ) - 1
              val = val + prob%A%val( l )
            END DO
            B( i ) = val
          END DO
        END IF

!  set the right-hand side according to supplied data

      ELSE
        IF ( transpose ) THEN
          B( : n ) = prob%G( : n ) + one
          B( n + 1 : n_total ) = one
        ELSE
          B = prob%C + one
        END IF
      END IF
!write(6,"( 'g', /, ( 5ES12.4 ) )" ) B( : n_total )

!  Call the factorization package

      IF ( do_solve .AND. prob%n > 0 ) THEN

!  if required, compute the preconditioner

        CALL CPU_TIME( times ) ; CALL CLOCK_time( clocks )

        SELECT CASE ( preconditioner )

!  form the diagonal preconditioner

        CASE ( diagonal_preconditioner )
          ALLOCATE( D( n_used ), STAT = alloc_stat )
          IF ( alloc_stat /= 0 ) THEN
            IF ( printe ) WRITE( out, 2150 ) 'X, V, U, RES', alloc_stat
            STOP
          END IF

          IF ( transpose ) THEN
            IF ( printo ) WRITE( out,                                          &
              "( /, ' ** diagonal preconditioner used on transpose **' )" )
            DO i = 1, m
              val = zero
              DO l = prob%A%ptr( i ), prob%A%ptr( i + 1 ) - 1
                j = prob%A%col( l )
                val = val + prob%A%val( l ) ** 2
              END DO
              D( i ) = val
            END DO
          ELSE
            IF ( printo ) WRITE( out,                                          &
              "( /, ' ** diagonal preconditioner used **')")
            D = zero
            DO i = 1, m
              DO l = prob%A%ptr( i ), prob%A%ptr( i + 1 ) - 1
                j = prob%A%col( l )
                D( j ) = D( j ) + prob%A%val( l ) ** 2
              END DO
            END DO
          END IF
          D = MAX( SQRT( D ), d_min )
          n_fact = n_used

!  form the MIQR preconditioner R^T R

        CASE ( miqr_preconditioner )
          CALL MIQR_initialize( MIQR_data, MIQR_control, MIQR_inform )
          IF ( is_specfile )                                                   &
            CALL MIQR_read_specfile( MIQR_control, input_specfile )
          IF ( transpose ) THEN
            IF ( printo ) WRITE( out,                                          &
              "( /, ' ** MIQR preconditioner used on transpose **' )" )
          ELSE
            IF ( printo ) WRITE( out, "( /, ' ** MIQR preconditioner used **')")
          END IF

          MIQR_control%transpose = transpose
          CALL MIQR_form( prob%A, MIQR_data, MIQR_control, MIQR_inform )
          IF ( printe ) WRITE( out, "( ' on exit from MIQR_form, status = ',   &
         &  I0, /, ' entries = ', I0, ', dropped = ', I0, ', time = ', F6.2 )")&
            MIQR_inform%status, MIQR_inform%entries_in_factors,                &
            MIQR_inform%drop, MIQR_inform%time%clock_form
          IF ( printo .AND. MIQR_inform%zero_diagonals > 0 )                   &
            WRITE( out, "( 1X, I0, ' zero diagonals' )" )                      &
              MIQR_inform%zero_diagonals
          n_fact = MIQR_inform%entries_in_factors

          IF ( MIQR_inform%status /= GALAHAD_ok ) THEN
            status = MIQR_inform%status
            GO TO 500
          END IF

!  form the MI35 preconditioner

        CASE ( mi35_preconditioner )

!  convert to column format

          CALL CONVERT_to_sparse_column_format( prob%A, A_by_cols,             &
                                                CONVERT_control,               &
                                                CONVERT_inform, IW, W )

!  if required, record the transpose

          IF ( transpose ) THEN
            ALLOCATE( A_ls%ptr( prob%A%m + 1 ),                                &
                      A_ls%row( prob%A%ptr( prob%A%m + 1 ) - 1 ),              &
                      A_ls%val( prob%A%ptr( prob%A%m + 1 ) - 1 ) )
            CALL CONVERT_transpose( A_by_cols%m, A_by_cols%n, A_by_cols%ne,    &
                                    A_by_cols%ptr, A_by_cols%row,              &
                                    A_by_cols%val, A_ls%ptr, A_ls%row, A_ls%val)
            A_ls%m = A_by_cols%n ; A_ls%n = A_by_cols%m
          ELSE
            A_ls%m = prob%A%m ; A_ls%n = prob%A%n
            ALLOCATE( A_ls%ptr( prob%A%n + 1 ),                                &
                      A_ls%row( prob%A%ptr( prob%A%n + 1 ) - 1 ),              &
                      A_ls%val( prob%A%ptr( prob%A%n + 1 ) - 1 ) )
            A_ls%ptr( : prob%A%n + 1 ) = A_by_cols%ptr( : prob%A%n + 1 )
            A_ls%row( : prob%A%ne ) = A_by_cols%row( : prob%A%ne )
            A_ls%val( : prob%A%ne ) = A_by_cols%val( : prob%A%ne )
          END IF
write(6,*) ' m, n, = ', A_ls%m, A_ls%n

!  check the matrix A, remove null rows/cols and adjust B accordingly

!write(19,*) prob%A%m, prob%A%n
!write(19,*)  prob%A%ptr( : prob%A%n + 1 )
!write(19,*)  prob%A%row( :  prob%A%ptr( prob%A%n + 1 ) - 1 )
!write(19,*)  prob%A%val( :  prob%A%ptr( prob%A%n + 1 ) - 1 )

!         control_mi35%limit_rowA = MAX( 100, A_ls%m / 1000 )
          control_mi35%limit_rowA = 3000
          CALL mi35_check_matrix( A_ls%m, A_ls%n, A_ls%ptr,                    &
                                  A_ls%row, A_ls%val,                          &
                                  control_mi35, info_mi35, b = B )
write(6,*) ' m, n, = ', A_ls%m, A_ls%n, ' check flag ', info_mi35%flag

          IF ( info_mi35%flag < 0 ) THEN
            status = GALAHAD_error_mi35
            IF ( printe ) WRITE( out, "( ' on exit from mi35_check_matrix, ',  &
           & 'status = ', I0 )" ) info_mi35%flag
            GO TO 500
          END IF

!  input (cleaned) matrix A; compute incomplete factorization of C = A^T*A.

!         size_l = 1 ; size_r = 0
!         size_l = 10 ; size_r = 10
!         size_l = 20 ; size_r = 20
          size_l = mi35_size_l ; size_r = mi35_size_r
!         size_l = A_ls%m ; size_r = A_ls%m

          control_mi35%iorder = 6
          control_mi35%iscale = mi35_scale
!         control_mi35%limit_colC = MAX( 100, A_ls%m / 1000 )
          control_mi35%limit_colC = - 1
          control_mi35%limit_C = - 1
          control_mi35%tau1 = mi35_tau1 ; control_mi35%tau2 = mi35_tau2

!write(6,*) ' max col = ', control_mi35%limit_colC


          CALL mi35_factorize( A_ls%m, A_ls%n, A_ls%ptr,                       &
                               A_ls%row, A_ls%val, size_l, size_r,             &
                               keep_mi35, control_mi35, info_mi35 )
!                              scale_mi35, perm_mi35 )

!write(6,*) 'control ', control_mi35
!write(6,*) 'info ', info_mi35


          CALL CPU_TIME( timep ) ; CALL CLOCK_time( clockp )
          n_fact = keep_mi35%fact_ptr( A_ls%n + 1 ) - 1
          IF ( printe ) WRITE( out, "( ' on exit from mi35_factorize, ',       &
         & 'status = ', I0, ', entries in L = ', I0, ', time = ', F6.2 )")     &
              info_mi35%flag, n_fact, clockp - clocks

          IF ( info_mi35%flag < 0 ) THEN
            status = GALAHAD_error_mi35
            GO TO 500
          END IF

          ALLOCATE( RHS( n_used ), STAT = alloc_stat )

!  form the MI35 (with C) preconditioner

        CASE ( mi35_with_c_preconditioner )

!  convert to column format

          CALL CONVERT_to_sparse_column_format( prob%A, A_by_cols,             &
                                                CONVERT_control,               &
                                                CONVERT_inform, IW, W )

!  if required, record the transpose

          IF ( transpose ) THEN
            ALLOCATE( A_ls%ptr( prob%A%m + 1 ),                                &
                      A_ls%row( prob%A%ptr( prob%A%m + 1 ) - 1 ),              &
                      A_ls%val( prob%A%ptr( prob%A%m + 1 ) - 1 ) )
            CALL CONVERT_transpose( A_by_cols%m, A_by_cols%n, A_by_cols%ne,    &
                                    A_by_cols%ptr, A_by_cols%row,              &
                                    A_by_cols%val, A_ls%ptr, A_ls%row, A_ls%val)
            A_ls%m = A_by_cols%n ; A_ls%n = A_by_cols%m
          ELSE
            A_ls%m = prob%A%m ; A_ls%n = prob%A%n
            ALLOCATE( A_ls%ptr( prob%A%n + 1 ),                                &
                      A_ls%row( prob%A%ptr( prob%A%n + 1 ) - 1 ),              &
                      A_ls%val( prob%A%ptr( prob%A%n + 1 ) - 1 ) )
            A_ls%ptr( : prob%A%n + 1 ) = A_by_cols%ptr( : prob%A%n + 1 )
            A_ls%row( : prob%A%ne ) = A_by_cols%row( : prob%A%ne )
            A_ls%val( : prob%A%ne ) = A_by_cols%val( : prob%A%ne )
          END IF

write(6,*) ' m, n, = ', A_ls%m, A_ls%n

!open(61)
!write(61,*)  A_ls%n, A_LS%ptr( A_ls%n + 1 ) - 1
!do i = 1, A_ls%n
! write(61,*) A_LS%ptr( i ), A_LS%ptr( i + 1 ) - 1
!  do l = A_LS%ptr( i ), A_LS%ptr( i + 1 ) - 1
!    write(61, * ) A_LS%row( l ), A_LS%val( l )
!  end do
!end do
!close(61)

!  check the matrix A, remove null rows/cols and adjust B accordingly

!         control_mi35%limit_rowA = MAX( 100, A_ls%m / 1000 )
          control_mi35%limit_rowA = 3000
          CALL mi35_check_matrix( A_ls%m, A_ls%n, A_ls%ptr, A_ls%row, A_ls%val,&
                                  control_mi35, info_mi35, b = B )
write(6,*) ' m, n, = ', A_ls%m, A_ls%n, ' check flag ', info_mi35%flag

          IF ( info_mi35%flag < 0 ) THEN
            status = GALAHAD_error_mi35
            IF ( printe ) WRITE( out, "( ' on exit from mi35_check_matrix, ',  &
           & 'status = ', I0 )" ) info_mi35%flag
            GO TO 500
          END IF

! form S = A^T*A  using cleaned A

          ALLOCATE( S%ptr( A_ls%n + 1 ), STAT = alloc_stat )
          CALL mi35_formC( A_ls%m, A_ls%n, A_ls%ptr, A_ls%row,                 &
                           A_ls%val, S%ptr, S%row, S%val, control_mi35,        &
                           info_mi35 )

!do i = 1, A_ls%n
! write(66,*) ' column ', i, ' nz = ', S%ptr( i + 1 ) - S%ptr( i ) - 1
!  do l = S%ptr( i ), S%ptr( i + 1 ) - 1
!    write(66,"( 2I7, ES12.4 )" ) S%row( l ), i, S%val( l )
!  end do
!end do

          CALL CPU_TIME( timep ) ; CALL CLOCK_time( clockp )
          n_fact = S%ptr( A_ls%n + 1 ) - 1
          IF ( printe ) WRITE( out, "( ' on exit from mi35_formC, ',           &
         & 'status = ', I0, ', entries in C = ', I0, ', time = ', F6.2 )")     &
              info_mi35%flag, n_fact, clockp - clocks

          IF ( info_mi35%flag < 0 ) THEN
            status = GALAHAD_error_mi35
            IF ( printe ) WRITE( out, "( ' on exit from mi35_formC, ',         &
           & 'status = ', I0 )" ) info_mi35%flag
            GO TO 500
          END IF

!  input (cleaned) matrix A; compute incomplete factorization of C = A^T*A.

!         size_l = 1 ; size_r = 0
!         size_l = 10 ; size_r = 10
!         size_l = 20 ; size_r = 20
          size_l = mi35_size_l ; size_r = mi35_size_r
!         size_l = A_ls%m ; size_r = A_ls%m

!         control_mi35%iorder = 0
!         control_mi35%iscale = 0
!         control_mi35%limit_colC = MAX( 100, A_ls%m / 1000 )
          control_mi35%limit_colC = - 1
          control_mi35%limit_C = - 1
          control_mi35%tau1 = mi35_tau1 ; control_mi35%tau2 = mi35_tau2

          CALL mi35_factorizeC( A_ls%n, S%ptr, S%row, S%val, size_l, size_r,   &
                               keep_mi35, control_mi35, info_mi35 )
!                              scale_mi35, perm_mi35 )

          CALL CPU_TIME( timep ) ; CALL CLOCK_time( clockp )
          n_fact = keep_mi35%fact_ptr( A_ls%n + 1 ) - 1
          IF ( printe ) WRITE( out, "( ' on exit from mi35_factorize, ',       &
         & 'status = ', I0, ', entries in L = ', I0, ', time = ', F6.2 )")     &
              info_mi35%flag, n_fact, clockp - clocks

          IF ( info_mi35%flag < 0 ) THEN
            status = GALAHAD_error_mi35
            GO TO 500
          END IF

!do i = 1, m
! write(67,*) ' column ', i, ' nz = ', S%ptr( i + 1 ) - S%ptr( i ) -1
!  do l = keep_mi35%fact_ptr( i ), keep_mi35%fact_ptr( i + 1 ) - 1
!    write(67,"( 2I7, ES12.4 )" ) keep_mi35%fact_row( l ), i,  keep_mi35%fact_val( l )
!  end do
!end do

          ALLOCATE( RHS( n_used ), STAT = alloc_stat )

        CASE DEFAULT
          n_fact = 0
        END SELECT

        DEALLOCATE( IW )

        CALL CPU_TIME( timep ) ; CALL CLOCK_time( clockp )
        IF ( printe ) WRITE( out, "( ' preconditioning time = ', F6.2 )" )     &
          clockp - clocks

!  ================
!  test the factors
!  ================

!       solv = ' LSTR'
!       IF ( printo ) WRITE( out, " ( ' ** LSTR solver used ** ' ) " )

        IF ( .FALSE. ) THEN ! test turned off

!  Form u = A * R^-1 * w with w = e_n

          W( : n_used ) = zero ; W( n_used ) = one
!write(6,*) n_used

          SELECT CASE ( preconditioner )
          CASE ( diagonal_preconditioner )
            W( : n_used ) = W( : n_used ) / D( : n_used )
          CASE ( miqr_preconditioner )
            CALL MIQR_apply( W, .FALSE., MIQR_data, MIQR_inform )
            write(6,*) ' norm R^-1 w  ', TWO_NORM( W( : n_used ) )
          CASE ( mi35_preconditioner, mi35_with_c_preconditioner )
            RHS( : n_used ) = W( : n_used )
            CALL mi35_solve( .FALSE., m, keep_mi35, RHS, W, info_mi35 )
          CASE DEFAULT
          END SELECT

          IF ( transpose ) THEN
            U = zero
            DO i = 1, m
              val = W( i )
              DO l = prob%A%ptr( i ), prob%A%ptr( i + 1 ) - 1
                j = prob%A%col( l )
                U( j ) = U( j ) + prob%A%val( l ) * val
              END DO
            END DO
          ELSE
            DO i = 1, m
              val = zero
              DO l = prob%A%ptr( i ), prob%A%ptr( i + 1 ) - 1
                val = val + prob%A%val( l ) * W( prob%A%col( l ) )
              END DO
              U( i ) = val
            END DO
          END IF

          write(6,*) ' norm sol ', TWO_NORM( U( : m_used ) )

!  Form w <- R^-T * A^T * u with u = e_m

          U( : m_used ) = zero ; U( m_used ) = one
          IF ( transpose ) THEN
            DO i = 1, m
              val = zero
              DO l = prob%A%ptr( i ), prob%A%ptr( i + 1 ) - 1
                val = val + prob%A%val( l ) * U( prob%A%col( l ) )
              END DO
              W( i ) = val
            END DO
          ELSE
            W( : n_used ) = zero
            DO i = 1, m
              val = U( i )
              DO l = prob%A%ptr( i ), prob%A%ptr( i + 1 ) - 1
                j = prob%A%col( l )
                W( j ) = W( j ) + prob%A%val( l ) * val
              END DO
            END DO
          END IF
          write(6,*) ' norm A^T x ', TWO_NORM( W( : n_used ) )

          SELECT CASE ( preconditioner )
          CASE ( diagonal_preconditioner )
            W( : n_used ) = W( : n_used ) / D( : n_used )
          CASE ( miqr_preconditioner )
            CALL MIQR_apply( W, .TRUE., MIQR_data, MIQR_inform )
          CASE ( mi35_preconditioner, mi35_with_c_preconditioner )
            RHS( : n_used ) = W( : n_used )
            CALL mi35_solve( .TRUE., m, keep_mi35, RHS, W, info_mi35 )
          CASE DEFAULT
          END SELECT

          write(6,*) ' norm sol ', TWO_NORM( W( : n_used ) )
        END IF

!  =================
!  solve the problem
!  =================

!  special case for mi35 derivatives
!  ---------------------------------

        IF ( preconditioner == mi35_preconditioner .OR.                        &
             preconditioner == mi35_with_c_preconditioner ) THEN

!  Iteration to find the minimizer

          LSTR_inform%status = 1
          U( : A_ls%m ) = B( : A_ls%m )

          DO
            CALL LSTR_solve( A_ls%m, A_ls%n, radius, X( : A_ls%n ),            &
                             U( : A_ls%m ), V( : A_ls%n ), LSTR_data,          &
                             LSTR_control, LSTR_inform )
            branch = LSTR_inform%status

!  Branch as a result of inform%status

            SELECT CASE( branch )

!  Form u <- u + A * R^-1 * v

            CASE( 2 )

              W( : A_ls%n ) = V( : A_ls%n )
              RHS( : A_ls%n ) = W( : A_ls%n )

              CALL mi35_solve( .TRUE., A_ls%n, keep_mi35, RHS, W, info_mi35 )

              DO j = 1, A_ls%n
                val = W( j )
                DO l = A_ls%ptr( j ), A_ls%ptr( j + 1 ) - 1
                  i = A_ls%row( l )
                  U( i ) = U( i ) + A_ls%val( l ) * val
                END DO
              END DO

!  Form v <- v + R^-T * A^T * u

            CASE( 3 )

              DO j = 1, A_ls%n
                val = zero
                DO l = A_ls%ptr( j ), A_ls%ptr( j + 1 ) - 1
                  val = val + A_ls%val( l ) * U( A_ls%row( l ) )
                END DO
                W( j ) = val
              END DO
              RHS( : A_ls%n ) = W( : A_ls%n )

              CALL mi35_solve( .FALSE., A_ls%n, keep_mi35, RHS, W, info_mi35 )

              V( : A_ls%n ) = V( : A_ls%n ) + W( : A_ls%n )

!  Restart

            CASE ( 4 )
              U( : A_ls%m ) = B( : A_ls%m )

!  Successful return

            CASE ( - 30, 0 )

!  recover x <- R^-1 x

              x_norm = SQRT( DOT_PRODUCT( X( : A_ls%n ) , X( : A_ls%n ) ) )
              RHS( : A_ls%n ) = X( : A_ls%n )

              CALL mi35_solve( .TRUE., A_ls%n, keep_mi35, RHS, X, info_mi35 )

!  Compute the residuals for checking

              RES( : A_ls%m ) = B( : A_ls%m )
              DO j = 1, A_ls%n
                val = X( j )
                DO l = A_ls%ptr( j ), A_ls%ptr( j + 1 ) - 1
                  i = A_ls%row( l )
                  RES( i ) = RES( i ) - A_ls%val( l ) * val
                END DO
              END DO

              CALL CPU_TIME( timet ) ; CALL CLOCK_time( clockt )
              IF ( printe ) THEN
                IF ( transpose ) THEN
                  WRITE( out, "( ' problem: ', A, ' (transposed), m = ', I0,   &
                & ', n = ', I0 )" ) TRIM( pname ), n_total, m
                ELSE
                  WRITE( out, "( ' problem: ', A, ', m = ', I0, ', n = ', I0)")&
                    TRIM( pname ), m, n_total
                END IF
                WRITE( out, "( 1X, I0, ' 1st pass and ', I0,                   &
               &  ' 2nd pass LSTR iterations, overall time = ', F6.2 )" )      &
                  LSTR_inform%iter, LSTR_inform%iter_pass2, clockt - clocks
                WRITE( out, "( '  ||b||   calculated              =',          &
               &  ES15.8 )" ) SQRT( DOT_PRODUCT( B( : A_ls%m ), B( : A_ls%m)))
                WRITE( out, "( '  ||x||_P recurred and calculated =',          &
               &  2ES15.8 )" ) LSTR_inform%x_norm, x_norm
                WRITE( out, "( ' ||Ax-b|| recurred and calculated =',          &
               &  2ES15.8 )" ) LSTR_inform%r_norm,                             &
                 SQRT( DOT_PRODUCT( RES( : A_ls%m ), RES( : A_ls%m ) ) )
              END IF
              EXIT

!  Error returns

            CASE DEFAULT
              CALL CPU_TIME( timet ) ; CALL CLOCK_time( clockt )
              IF ( printe ) WRITE( out, "( ' LSTR_solve exit status = ', I0 )")&
                LSTR_inform%status
              EXIT

           END SELECT
          END DO

!  usual preconditioners
!  ---------------------

        ELSE

!  Iteration to find the minimizer

          LSTR_inform%status = 1
          U = B

          DO
            CALL LSTR_solve( m_used, n_used, radius, X,  U, V, LSTR_data,      &
                             LSTR_control, LSTR_inform )
            branch = LSTR_inform%status

!  Branch as a result of inform%status

            SELECT CASE( branch )

!  Form u <- u + A * R^-1 * v

            CASE( 2 )
              W( : n_used ) = V( : n_used )

              SELECT CASE ( preconditioner )
              CASE ( diagonal_preconditioner )
                W( : n_used ) = W( : n_used ) / D( : n_used )
              CASE ( miqr_preconditioner )
                CALL MIQR_apply( W, .FALSE., MIQR_data, MIQR_inform )
              CASE DEFAULT
              END SELECT

              IF ( transpose ) THEN
                DO i = 1, m
                  val = W( i )
                  DO l = prob%A%ptr( i ), prob%A%ptr( i + 1 ) - 1
                    j = prob%A%col( l )
                    U( j ) = U( j ) + prob%A%val( l ) * val
                  END DO
                END DO
              ELSE
                DO i = 1, m
                  val = U( i )
                  DO l = prob%A%ptr( i ), prob%A%ptr( i + 1 ) - 1
                    val = val + prob%A%val( l ) * W( prob%A%col( l ) )
                  END DO
                  U( i ) = val
                END DO
              END IF

!  Form v <- v + R^-T * A^T * u

            CASE( 3 )
              IF ( transpose ) THEN
                DO i = 1, m
                  val = zero
                  DO l = prob%A%ptr( i ), prob%A%ptr( i + 1 ) - 1
                    val = val + prob%A%val( l ) * U( prob%A%col( l ) )
                  END DO
                  W( i ) = val
                END DO
              ELSE
                W( : n_used ) = zero
                DO i = 1, m
                  val = U( i )
                  DO l = prob%A%ptr( i ), prob%A%ptr( i + 1 ) - 1
                    j = prob%A%col( l )
                    W( j ) = W( j ) + prob%A%val( l ) * val
                  END DO
                END DO
              END IF

              SELECT CASE ( preconditioner )
              CASE ( diagonal_preconditioner )
                W( : n_used ) = W( : n_used ) / D( : n_used )
              CASE ( miqr_preconditioner )
                CALL MIQR_apply( W, .TRUE., MIQR_data, MIQR_inform )
              CASE DEFAULT
              END SELECT

              V( : n_used ) = V( : n_used ) + W( : n_used )

!  Restart

            CASE ( 4 )
              U = B

!  Successful return

            CASE ( - 30, 0 )

!  recover x <- R^-1 x

              x_norm = SQRT( DOT_PRODUCT( X, X ) )

              SELECT CASE ( preconditioner )
              CASE ( diagonal_preconditioner )
                X( : n_used ) = X( : n_used ) / D( : n_used )
              CASE ( miqr_preconditioner )
                CALL MIQR_apply( X, .FALSE., MIQR_data, MIQR_inform )
              CASE DEFAULT
              END SELECT

!  Compute the residuals for checking

              IF ( transpose ) THEN
                RES = B
                DO i = 1, m
                  val = X( i )
                  DO l = prob%A%ptr( i ), prob%A%ptr( i + 1 ) - 1
                    j = prob%A%col( l )
                    RES( j ) = RES( j ) - prob%A%val( l ) * val
                  END DO
                END DO
              ELSE
                DO i = 1, m
                  val = B( i )
                  DO l = prob%A%ptr( i ), prob%A%ptr( i + 1 ) - 1
                    val = val - prob%A%val( l ) * X( prob%A%col( l ) )
                  END DO
                  RES( i ) = val
                END DO
              END IF

              CALL CPU_TIME( timet ) ; CALL CLOCK_time( clockt )
              IF ( printe ) THEN
                IF ( transpose ) THEN
                  WRITE( out, "( ' problem: ', A, ' (transposed), m = ', I0,   &
                & ', n = ', I0 )" ) TRIM( pname ), n_total, m
                ELSE
                  WRITE( out, "( ' problem: ', A, ', m = ', I0, ', n = ', I0)")&
                    TRIM( pname ), m, n_total
                END IF
                WRITE( out, "( 1X, I0, ' 1st pass and ', I0,                   &
               &  ' 2nd pass LSTR iterations, overall time = ', F6.2 )" )      &
                  LSTR_inform%iter, LSTR_inform%iter_pass2, clockt - clocks
                WRITE( out, "( '  ||b||   calculated              =',          &
               &  ES15.8 )" ) SQRT( DOT_PRODUCT( B, B ) )
                WRITE( out, "( '  ||x||_P recurred and calculated =',          &
               &  2ES15.8 )" ) LSTR_inform%x_norm, x_norm
                WRITE( out, "( ' ||Ax-b|| recurred and calculated =',          &
               &  2ES15.8 )" ) LSTR_inform%r_norm,                             &
                  SQRT( DOT_PRODUCT( RES, RES))
                END IF
              EXIT

!  Error returns

            CASE DEFAULT
              CALL CPU_TIME( timet ) ; CALL CLOCK_time( clockt )
              IF ( printe ) WRITE( out, "( ' LSTR_solve exit status = ', I0 )")&
                LSTR_inform%status
              EXIT
            END SELECT
          END DO
        END IF
      END IF

      SELECT CASE ( preconditioner )
      CASE( diagonal_preconditioner )
        IF ( printe ) WRITE( out, "( ' diagonal preconditioner' )" )
      CASE( miqr_preconditioner )
        IF ( printe ) WRITE( out, "( ' MIQR preconditioner (level<=',I0,       &
       & ',order<=',I0,',fill_r<=',I0,',fill_q<=',I0,',',/,22X,'av_r=',F0.1,   &
       &',av_q=', F0.1,')' )" ) MIQR_control%max_level, MIQR_control%max_order,&
         MIQR_control%max_fill, MIQR_control%max_fill_q,                       &
         MIQR_control%average_max_fill, MIQR_control%average_max_fill_q
      CASE( mi35_preconditioner )
        IF ( printe ) WRITE( out, "( ' MI35 preconditioner ',                  &
       & '(l=',I0,',r=',I0,',scale=',I0,',tau1=',F0.1,',tau2=',F0.1,')' )" )   &
            mi35_size_l, mi35_size_r, mi35_scale, mi35_tau1, mi35_tau2
      CASE( mi35_with_c_preconditioner )
        IF ( printe ) WRITE( out, "( ' MI35 (with C) preconditioner ',         &
       & '(l=',I0,',r=',I0,',scale=',I0,',tau1=',F0.1,',tau2=',F0.1,')' )" )   &
            mi35_size_l, mi35_size_r, mi35_scale, mi35_tau1, mi35_tau2
      CASE DEFAULT
        IF ( printe ) WRITE( out, "( ' no preconditioner' )" )
      END SELECT

      IF ( write_result_summary ) THEN
        BACKSPACE( rfiledevice )
!       WRITE( rfiledevice, 2190 )                                             &
!          pname, n, m, iter, qfval, status, clockt
        IF ( LSTR_inform%status >= 0 ) THEN
          WRITE( rfiledevice, "( A10, 2I8, I3, L2, I10, I8, 2F12.2, I4 )" )    &
            pname, m_used, n_used, preconditioner, consistent, n_fact,         &
            LSTR_inform%iter, clockp - clocks,                                 &
            clockt - clocks, LSTR_inform%status
        ELSE
          WRITE( rfiledevice, "( A10, 2I8, I3, L2, I10, I8, 2F12.2, I4 )" )    &
            pname, m_used, n_used, preconditioner, consistent, - n_fact,       &
            - LSTR_inform%iter,                                                &
            - ( clockp - clocks ), - ( clockt - clocks ), - LSTR_inform%status
        END IF
      END IF
      CALL LSTR_terminate( LSTR_data, LSTR_control, LSTR_inform )
      GO TO 600

!  error returns

  500 CONTINUE
      IF ( write_result_summary ) THEN
        BACKSPACE( rfiledevice )
        CALL CPU_TIME( timet ) ; CALL CLOCK_time( clockt )
        WRITE( rfiledevice, "( A10, 2I8, I3, L2, 2I8, 2F12.2, I4 )" )          &
          pname, m_used, n_used, preconditioner, consistent, - n_fact,         &
          - 1, - ( clockt - clocks ), - ( clockt - clocks ), - status
      END IF

 ! delete internal workspace

  600 CONTINUE
      SELECT CASE ( preconditioner )
      CASE ( diagonal_preconditioner )
        DEALLOCATE( D, STAT = alloc_stat )
      CASE ( miqr_preconditioner )
        CALL MIQR_terminate( MIQR_data, MIQR_control, MIQR_inform )
      CASE ( mi35_preconditioner )
        DEALLOCATE( RHS, STAT = alloc_stat )
      CASE ( mi35_with_c_preconditioner )
        DEALLOCATE( RHS, S%ptr, S%row, S%col, S%val, STAT = alloc_stat )
      CASE DEFAULT
      END SELECT

      DEALLOCATE( X, V, W, B, U, RES, STAT = alloc_stat )
      IF ( is_specfile ) CLOSE( input_specfile )

      CALL CUTEST_cterminate( cutest_status )
      RETURN

 910  CONTINUE
      WRITE( out, "( ' CUTEst error, status = ', i0, ', stopping' )" )         &
        cutest_status
      status = - 98
      RETURN

!  Non-executable statements

 2150 FORMAT( ' Allocation error, variable ', A8, ' status = ', I0 )
 2160 FORMAT( ' IOSTAT = ', I6, ' when opening file ', A9, '. Stopping ' )
 2180 FORMAT( A10 )
 2200 FORMAT( /, ' problem dimensions:  n = ', I0, ', slacks = ', I0,          &
                 ', m = ', I0, ', a_ne = ', I0 )

!  End of subroutine USE_LSTR

     END SUBROUTINE USE_LSTR

!  End of module USELSTR_double

   END MODULE GALAHAD_USELSTR_double
