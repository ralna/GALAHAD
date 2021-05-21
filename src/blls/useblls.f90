! THIS VERSION: GALAHAD 3.3 - 20/05/2021 AT 11:00 GMT.

!-*-*-*-*-*-*-*-  G A L A H A D   U S E B L L S   M O D U L E  -*-*-*-*-*-*-*-*-

!  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
!  Principal authors: Nick Gould and Dominique Orban

!  History -
!   originally released with GALAHAD Version 3.3. November 27th 2020

!  For full documentation, see
!   http://galahad.rl.ac.uk/galahad-www/specs.html

    MODULE GALAHAD_USEBLLS_double

!     -----------------------------------------------------
!    | CUTEst/AMPL interface to BLLS, a preconditioned     |
!    | projected conjugate-gradient algorithm for          |
!    | bound-constrained linear least-squares minimization |
!     -----------------------------------------------------

      USE CUTEst_interface_double
!NOT95USE GALAHAD_CPU_time
      USE GALAHAD_QPT_double
      USE GALAHAD_BLLS_double
      USE GALAHAD_SPECFILE_double
      USE GALAHAD_COPYRIGHT
!     USE GALAHAD_SCALING_double
      USE GALAHAD_SYMBOLS

      IMPLICIT NONE

      PRIVATE
      PUBLIC :: USE_BLLS

    CONTAINS

!-*-*-*-*-*-*-*-*-*-   U S E _ B L L S  S U B R O U T I N E   -*-*-*-*-*-*-*-*-

     SUBROUTINE USE_BLLS( input )

!  --------------------------------------------------------------------
!
!  Solve the bound-constrained linear-least squares problem
!
!      minimize     1/2 || A x - b ||_2^2
!
!     subject to      x_l <= x <= x_u
!
!  using the GALAHAD package GALAHAD_BLLS
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

      INTEGER :: n, m, ir, ic, iores, smt_stat, cutest_status
      INTEGER :: i, j, k, l, nfixed, ndegen, alloc_stat, nnzj, n_s, blls_status
      INTEGER :: e_order = 0
      INTEGER :: l_order = 0
      INTEGER :: v_order = 0
!     INTEGER :: np1, npm
!     INTEGER :: factorization_integer, factorization_real
      REAL :: time, timeo, times, timet
      REAL ( KIND = wp ) :: stopr
      LOGICAL :: filexx, printo, printe, is_specfile
!     LOGICAL :: ldummy

!  Specfile characteristics

      INTEGER, PARAMETER :: input_specfile = 34
      INTEGER, PARAMETER :: lspec = 21
      CHARACTER ( LEN = 16 ) :: specname = 'RUNBLLS'
      TYPE ( SPECFILE_item_type ), DIMENSION( lspec ) :: spec
      CHARACTER ( LEN = 16 ) :: runspec = 'RUNBLLS.SPC'

!  The default values for BLLS could have been set as:

! BEGIN RUNBLLS SPECIFICATIONS (DEFAULT)
!  write-problem-data                                NO
!  problem-data-file-name                            BLLS.data
!  problem-data-file-device                          26
!  write-initial-sif                                 NO
!  initial-sif-file-name                             INITIAL.SIF
!  initial-sif-file-device                           51
!  scale-problem                                     0
!  solve-problem                                     YES
!  print-full-solution                               NO
!  write-solution                                    NO
!  solution-file-name                                BLLSSOL.d
!  solution-file-device                              62
!  write-result-summary                              NO
!  result-summary-file-name                          BLLSRES.d
!  result-summary-file-device                        47
!  perturb-bounds-by                                 0.0
! END RUNBLLS SPECIFICATIONS

!  Default values for specfile-defined parameters

!     INTEGER :: scale = 0
      INTEGER :: dfiledevice = 26
      INTEGER :: ifiledevice = 51
      INTEGER :: rfiledevice = 47
      INTEGER :: sfiledevice = 62
      LOGICAL :: write_problem_data   = .FALSE.
      LOGICAL :: write_initial_sif    = .FALSE.
      LOGICAL :: write_solution       = .FALSE.
      LOGICAL :: write_result_summary = .FALSE.
      CHARACTER ( LEN = 30 ) :: dfilename = 'BLLS.data'
      CHARACTER ( LEN = 30 ) :: ifilename = 'INITIAL.SIF'
      CHARACTER ( LEN = 30 ) :: rfilename = 'BLLSRES.d'
      CHARACTER ( LEN = 30 ) :: sfilename = 'BLLSSOL.d'
!     LOGICAL :: do_solve = .TRUE.
      LOGICAL :: fulsol = .FALSE.
      REAL ( KIND = wp ) :: pert_bnd = zero

!  Output file characteristics

      INTEGER, PARAMETER :: out  = 6
      INTEGER, PARAMETER :: io_buffer = 11
      INTEGER :: errout = 6
      CHARACTER ( LEN =  5 ) :: state, solv
      CHARACTER ( LEN = 10 ) :: pname

!  Arrays

!     TYPE ( SCALING_control_type ) :: control
      TYPE ( BLLS_data_type ) :: data
      TYPE ( BLLS_control_type ) :: BLLS_control
      TYPE ( BLLS_inform_type ) :: BLLS_inform
      TYPE ( GALAHAD_userdata_type ) :: userdata
      TYPE ( QPT_problem_type ) :: prob

!  Allocatable arrays

      CHARACTER ( LEN = 10 ), ALLOCATABLE, DIMENSION( : ) :: VNAME, CNAME
!     REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: SH, SA
      REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: X, X_l, X_u
      REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: Y, C_l, C_u
      LOGICAL, ALLOCATABLE, DIMENSION( : ) :: EQUATN, LINEAR
      INTEGER, ALLOCATABLE, DIMENSION( : ) :: X_stat

      CALL CPU_TIME( time )

!  Determine the number of variables and constraints

      CALL CUTEST_cdimen( cutest_status, input, n, m )
      IF ( cutest_status /= 0 ) GO TO 910

!  allocate temporary arrays

      ALLOCATE( X( n ), X_l( n ), X_u( n ), Y( m ), C_l( m ), C_u( m ),        &
                EQUATN( m ), LINEAR( m ), STAT = alloc_stat )

      IF ( alloc_stat /= 0 ) THEN
        WRITE( out, 2150 ) 'X etc', alloc_stat ; STOP
      END IF

!  set up the data structures necessary to hold the group partially
!  separable function.

      CALL CUTEST_csetup( cutest_status, input, out,                           &
                          io_buffer, n, m, X, X_l, X_u, Y, C_l, C_u,           &
                          EQUATN, LINEAR, e_order, l_order, v_order )
      DEALLOCATE( Y, LINEAR, STAT = alloc_stat )

!  count the number of slack variables, and set problem dimensions

      n_s = m - COUNT( EQUATN )
      prob%m = m ; prob%n = n + n_s

!  Determine the names of the problem, variables and constraints.

      ALLOCATE( VNAME( prob%n ), CNAME( m ), STAT = alloc_stat )
      IF ( alloc_stat /= 0 ) THEN
        WRITE( out, 2150 ) 'VNAME etc', alloc_stat ; STOP
      END IF

      CALL CUTEST_cnames( cutest_status, n, m, pname, VNAME, CNAME )
      IF ( cutest_status /= 0 ) GO TO 910
      WRITE( out, "( /, ' Problem: ', A )" ) pname

!  allocate problem arrays

      ALLOCATE( prob%X( prob%n ), prob%X_l( prob%n ), prob%X_u( prob%n ),      &
                prob%B( prob%n ), prob%C( prob%m ), prob%Z( prob%n ),          &
                X_stat( prob%n ), STAT = alloc_stat )
      IF ( alloc_stat /= 0 ) THEN
        WRITE( out, 2150 ) 'prob%X etc', alloc_stat ; STOP
      END IF

!  transfer data to problem

      prob%X( : n ) = X( : n )
      prob%X_l( : n ) = X_l( : n )
      prob%X_u( : n ) = X_u( : n )
      DEALLOCATE( X_l, X_u, STAT = alloc_stat )

!  determine the number of entries in the Jacobian, and set its dimensions

      CALL CUTEST_cdimsj( cutest_status, nnzj )
      IF ( cutest_status /= 0 ) GO TO 910
      prob%A%m = prob%m ; prob%A%n = prob%n ; prob%A%ne = nnzj + n_s
      CALL SMT_put( prob%A%type, 'COORDINATE', smt_stat )

!  allocate problem arrays

      ALLOCATE( prob%A%val( prob%A%ne ), prob%A%row( prob%A%ne ),              &
                prob%A%col( prob%A%ne ), STAT = alloc_stat )
     IF ( alloc_stat /= 0 ) THEN
        WRITE( out, 2150 ) 'prob%A%val etc', alloc_stat ; STOP
     END IF

!  compute the values of the constraints and Jacobian

      CALL CUTEST_ccfsg( cutest_status, n, m, X, prob%C, nnzj, prob%A%ne,      &
                         prob%A%val, prob%A%col, prob%A%row, .TRUE. )
      prob%B = - prob%C
!write(6,*) ' count ', COUNT( prob%A%col( : nnzj ) == 0 )

!  deal with slack variables

      prob%A%ne = nnzj
      IF ( n_s > 0 ) THEN
        l = n
        DO i = 1, m
          IF ( .NOT. EQUATN( i ) ) THEN
            l = l + 1
            prob%X( l ) = zero
            prob%X_l( l ) = C_l( i )
            prob%X_u( l ) = C_u( i )
            VNAME( l ) = CNAME( i )
            prob%A%ne = prob%A%ne + 1
            prob%A%row( prob%A%ne ) = i
            prob%A%col( prob%A%ne ) = l
            prob%A%val( prob%A%ne ) = - one
          END IF
        END DO
      END IF
      DEALLOCATE( X, C_l, C_u, CNAME, EQUATN, STAT = alloc_stat )

!  ------------------- problem set-up complete ----------------------

      CALL CPU_TIME( times )

!  ------------------ Open the specfile for runblls ----------------

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
!       spec( 8 )%keyword = 'scale-problem'
!       spec( 13 )%keyword = 'solve-problem'
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
!       CALL SPECFILE_assign_integer( spec( 8 ), scale, errout )
!       CALL SPECFILE_assign_logical( spec( 13 ), do_solve, errout )
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

        n = prob%n ; k = prob%A%ne
        WRITE( dfiledevice, "( 'm, n = ', I0, 1X, I0 )" ) m, n
        WRITE( dfiledevice, "( ' b ', /, ( 5ES12.4 ) )" ) prob%B( : m )
        WRITE( dfiledevice, "( ' x_l ', /, ( 5ES12.4 ) )" ) prob%X_l( : n )
        WRITE( dfiledevice, "( ' x_u ', /, ( 5ES12.4 ) )" ) prob%X_u( : n )
        WRITE( dfiledevice, "( ' A_row ', /, ( 10I6 ) )" ) prob%A%row( : k )
        WRITE( dfiledevice, "( ' A_col ', /, ( 10I6 ) )" ) prob%A%col( : k )
        WRITE( dfiledevice, "( ' A_val ', /, ( 5ES12.4 ) )" ) prob%A%val( : k )

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
        WRITE( rfiledevice, "( A )" ) pname
      END IF

!  Set all default values, and override defaults if requested

!     CALL SCALING_initialize( control )

      CALL BLLS_initialize( data, BLLS_control, BLLS_inform )
      IF ( is_specfile )                                                       &
        CALL BLLS_read_specfile( BLLS_control, input_specfile )

!     control%print_level = BLLS_control%print_level
!     control%out         = BLLS_control%out
!     control%out_error   = BLLS_control%error

      printo = out > 0 .AND. BLLS_control%print_level > 0
      printe = out > 0 .AND. BLLS_control%print_level >= 0
      WRITE( out, "( /, ' problem dimensions:  m = ', I0, ', n = ', I0,        &
     &  ', a_ne = ', I0 )" ) prob%m, prob%n, prob%A%ne

      IF ( printo ) CALL COPYRIGHT( out, '2020' )
      X_stat = 0

! ! If required, scale the problem

!       IF ( scale > 0 ) THEN
!         ALLOCATE( SH( n ), SA( 0 ), STAT = alloc_stat )
!         IF ( alloc_stat /= 0 ) THEN
!           IF ( printe ) WRITE( out, 2150 ) 'SH/SA', alloc_stat ; STOP
!         END IF

! !  Scale using K

!         IF ( scale == 1 .OR. scale == 4 ) THEN
!           IF ( printo ) WRITE( out, 2140 ) 'K'
!           CALL SCALING_get_factors_from_K( n, 0, prob%H%val, prob%H%col,     &
!                                            prob%H%ptr, prob%A%val,prob%A%col,&
!                                            prob%A%ptr, SH, SA, control, ifail)
! !  Scale using A

!         ELSE IF ( scale == 2 .OR. scale == 5 ) THEN
!           IF ( printo ) WRITE( out, 2140 ) 'A'
!           CALL SCALING_get_factors_from_A( n, 0, prob%A%val, prob%A%col,     &
!                                            prob%A%ptr, SH, SA, control,ifail )
!         ELSE IF ( scale == 3 ) THEN
!           SH = one ; SA = one
!         END IF

! !  Rescale A

!         IF ( scale >= 3 ) THEN
!           IF ( printo ) WRITE( out, 2170 )
!           CALL SCALING_normalize_rows_of_A( n, 0, prob%A%val, prob%A%col,    &
!                                             prob%A%ptr, SH, SA )
!         END IF

! !  Apply the scaling factors

!         CALL SCALING_apply_factors( n, 0, prob%H%val, prob%H%col, prob%H%ptr,&
!                                     prob%A%val, prob%A%col, prob%A%ptr,      &
!                                     prob%G, prob%X, prob%X_l, prob%X_u,      &
!                                     prob%C_l, prob%C_u, prob%Y, prob%Z,      &
!                                     infinity, SH, SA, .TRUE. )
!       END IF

! !  If the preprocessor is to be used, or the problem to be output,
! !  allocate sufficient space

!       IF ( write_initial_sif ) THEN

!         ALLOCATE( prob%X_status( n ), STAT = alloc_stat )
!         IF ( alloc_stat /= 0 ) THEN
!           IF ( printe ) WRITE( out, 2150 ) 'X_status', alloc_stat
!           STOP
!         END IF
!         prob%X_status = ACTIVE

!         ALLOCATE( prob%Z_l( n ), prob%Z_u( n ), STAT = alloc_stat )
!         IF ( alloc_stat /= 0 ) THEN
!           IF ( printe ) WRITE( out, 2150 ) 'Z_lu', alloc_stat
!           STOP
!         END IF
!         prob%Z_l( : n ) = - infinity
!         prob%Z_u( : n ) =   infinity

! !  Writes the initial SIF file, if needed

!         IF ( write_initial_sif ) THEN
!           CALL QPT_write_to_sif( prob, pname, ifilename, ifiledevice,        &
!                                  .FALSE., .FALSE., infinity )
!           IF ( .NOT. do_solve ) STOP
!         END IF
!       END IF

!  Solve the problem

      CALL CPU_TIME( timeo )

      solv = ' BLLS'
      IF ( printo ) WRITE( out, " ( ' ** BLLS solver used ** ' ) " )
      CALL BLLS_solve( prob, X_stat, data, BLLS_control, BLLS_inform,          &
                       userdata )
      blls_status = BLLS_inform%status

      IF ( printo ) WRITE( out, " ( /, ' ** BLLS solver used ** ' ) " )
      CALL SYMBOLS_status( blls_status, out, ' ', 'BLLS_solve' )
      CALL CPU_TIME( timet )

!  Deallocate arrays from the minimization

      stopr = BLLS_control%stop_d
      CALL BLLS_terminate( data, BLLS_control, BLLS_inform )

!  If the problem was scaled, unscale it.
!
!      IF ( scale > 0 ) THEN
!        CALL SCALING_apply_factors( n, 0, prob%H%val, prob%H%col, prob%H%ptr, &
!                                    prob%A%val, prob%A%col, prob%A%ptr,       &
!                                    prob%G, prob%X, prob%X_l, prob%X_u,       &
!                                    prob%C_l, prob%C_u, prob%Y, prob%Z,       &
!                                    infinity, SH, SA, .FALSE., C = prob%C )
!        DEALLOCATE( SH, SA )
!      END IF

!  Print details of the solution obtained

      WRITE( out, 2010 ) blls_status
      IF ( blls_status == 0 .OR. blls_status == - 8 .OR.                       &
           blls_status == - 9 .OR. blls_status == - 10 ) THEN
        l = 4
        IF ( fulsol ) l = prob%n
        ir = 1 ; ic = MIN( l, prob%n )

!  Print details of the primal and dual variables

        WRITE( out, 2090 )
        DO j = 1, 2
          IF ( j == 2 ) THEN
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

        WRITE( out, "( /, ' Of the ', I0, ' variables, ', I0,                  &
       &                  ' are on bounds & ', I0, ' are dual degenerate' )" ) &
               prob%n, nfixed, ndegen

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

          WRITE( sfiledevice, "( /, ' Problem:    ', A10, /, ' Solver :   ',   &
         &       A5, /, ' Objective:', ES24.16 )" ) pname, solv, BLLS_inform%obj
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

          CLOSE( sfiledevice )
        END IF
      END IF

      times = times - time ; timet = timet - timeo
      WRITE( out, "( /, ' Total time = ', 0P, F12.2 )" ) times + timet
      WRITE( out, 2070 ) pname

!  Compare the variants used so far

      WRITE( out, "( A5, I7, 6X, ES12.4, I6, 0P, 3F8.2 )" ) solv,              &
                     BLLS_inform%iter, BLLS_inform%obj, blls_status,           &
                     times, timet, times + timet

      IF ( write_result_summary ) THEN
        BACKSPACE( rfiledevice )
        WRITE( rfiledevice, "( A10, 2I7, I6, ES13.4, I6, 0P, F8.2 )" ) pname,  &
           m, n, BLLS_inform%iter, BLLS_inform%obj, blls_status, timet
      END IF

      DEALLOCATE( prob%X, prob%X_l, prob%X_u, prob%B, prob%C, X_stat,          &
                prob%A%val, prob%A%row, prob%A%col, VNAME, STAT = alloc_stat )
      IF ( is_specfile ) CLOSE( input_specfile )

      CALL CUTEST_cterminate( cutest_status )
      RETURN

 910  CONTINUE
      WRITE( out, "( ' CUTEst error, status = ', i0, ', stopping' )" )         &
        cutest_status
      RETURN

!  Non-executable statements

 2000 FORMAT( '      . .          .....  ..........',                          &
              '  ..........  ..........  .......... ' )
 2010 FORMAT( /,' Stopping with inform%status = ', I3 )
 2050 FORMAT( I7, 1X, A10, A6, 4ES12.4 )
 2070 FORMAT( /, ' Problem: ', A, //,                                          &
                 '                     objective',                             &
                 '          < ------ time ----- > ', /,                        &
                 ' Method  iterations    value  ',                             &
                 '   status setup   solve   total', /,                         &
                 ' ------  ----------   -------   ',                           &
                 ' ------ -----    ----   -----  ' )
 2090 FORMAT( /, ' Solution : ', /, '                              ',          &
                 '        <------ Bounds ------> ', /                          &
                 '      # name       state    value   ',                       &
                 '    Lower       Upper       Dual ' )
! 2110 FORMAT( ' Of the ', I7, ' constraints, ', I7,' are equations &', I7,    &
!              ' are degenerate' )
!2140 FORMAT( /, ' *** Problem will be scaled based on ', A1, ' *** ' )
 2150 FORMAT( ' Allocation error, variable ', A8, ' status = ', I6 )
 2160 FORMAT( ' IOSTAT = ', I6, ' when opening file ', A9, '. Stopping ' )
!2170 FORMAT( /, ' *** Further scaling applied to A *** ' )

!  End of subroutine USE_BLLS

     END SUBROUTINE USE_BLLS

!  End of module USEBLLS_double

   END MODULE GALAHAD_USEBLLS_double
