! THIS VERSION: GALAHAD 2.5 - 07/02/2013 AT 14:00 GMT.

!-*-*-*-*-*-*-*-  G A L A H A D   U S E W C P   M O D U L E  -*-*-*-*-*-*-*-*-

!  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
!  Principal authors: Nick Gould and Dominique Orban

!  History -
!   originally released with GALAHAD Version 2.0. November 1st 2005

!  For full documentation, see
!   http://galahad.rl.ac.uk/galahad-www/specs.html

   MODULE GALAHAD_USEWCP_double

!    -------------------------------------------------------------
!    | CUTEst/AMPL interface to WCP, an interior-point algorithm |
!    | for finding a welll-centered point within a polytope      |
!    -------------------------------------------------------------

      USE CUTEst_interface_double
!NOT95USE GALAHAD_CPU_time
      USE GALAHAD_CLOCK
      USE GALAHAD_QPT_double
      USE GALAHAD_WCP_double
      USE GALAHAD_SORT_double, only: SORT_reorder_by_rows
      USE GALAHAD_PRESOLVE_double
      USE GALAHAD_SPECFILE_double 
      USE GALAHAD_COPYRIGHT
      USE GALAHAD_STRING_double
      USE GALAHAD_SYMBOLS,                                                     &
          ACTIVE                => GALAHAD_ACTIVE,                             &
          TRACE                 => GALAHAD_TRACE,                              &
          DEBUG                 => GALAHAD_DEBUG,                              &
          GENERAL               => GALAHAD_GENERAL,                            &
          ALL_ZEROS             => GALAHAD_ALL_ZEROS
      USE GALAHAD_SCALE_double

      IMPLICIT NONE

      PRIVATE
      PUBLIC :: USE_WCP

    CONTAINS

!-*-*-*-*-*-*-*-*-*-   U S E _ W C P  S U B R O U T I N E   -*-*-*-*-*-*-*-*-

     SUBROUTINE USE_WCP( input )

!  --------------------------------------------------------
!
!  Find a well-centerted point within the polytope
!
!         c_l <= A x <= c_u
!         x_l <=  x <= x_u
!
!  specified by CUTEst using the GALAHAD package GALAHAD_WCP
!
!  --------------------------------------------------------

!  Dummy argument

      INTEGER, INTENT( IN ) :: input

!  Parameters

      INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
      INTEGER, PARAMETER :: long = SELECTED_INT_KIND( 18 )
      INTEGER, PARAMETER :: io_buffer = 11
      REAL ( KIND = wp ), PARAMETER :: zero = 0.0_wp
      REAL ( KIND = wp ), PARAMETER :: one = 1.0_wp
      REAL ( KIND = wp ), PARAMETER :: ten = 10.0_wp
      REAL ( KIND = wp ), PARAMETER :: infinity = ten ** 19

!     INTEGER, PARAMETER :: n_k = 100, k_k = 3, in = 28
!     REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( :, : ) :: k_val
!     CHARACTER ( len = 10 ) :: filename = 'k.val'

!  Scalars

      INTEGER :: i, j, l, nea, n, m, ir, ic, la, liw, iores, cutest_status
!     INTEGER :: np1, npm
      INTEGER :: status, mfixed, mdegen, iter, nfacts, nfixed, ndegen, mequal
      INTEGER :: alloc_stat, A_ne, smt_stat
      INTEGER ( KIND = long ) :: factorization_integer, factorization_real
      REAL :: time, timeo, times, timet, timep1, timep2, timep3, timep4
      REAL ( KIND = wp ) :: clock, clocko, clocks, clockt
      REAL ( KIND = wp ) :: objf, qfval, stopr, dummy
      REAL ( KIND = wp ) :: res_c, res_k, max_cs
      LOGICAL :: filexx, printo, printe, is_specfile
!     LOGICAL :: ldummy
            
!  Functions

!$    INTEGER :: OMP_GET_MAX_THREADS

!  Specfile characteristics

      INTEGER, PARAMETER :: input_specfile = 34
      INTEGER, PARAMETER :: lspec = 26
      CHARACTER ( LEN = 16 ) :: specname = 'RUNWCP'
      TYPE ( SPECFILE_item_type ), DIMENSION( lspec ) :: spec
      CHARACTER ( LEN = 16 ) :: runspec = 'RUNWCP.SPC'

!  The default values for WCP could have been set as:

! BEGIN RUNWCP SPECIFICATIONS (DEFAULT)
!  write-problem-data                        NO
!  problem-data-file-name                    WCP.data
!  problem-data-file-device                  26
!  write-initial-sif                         NO
!  initial-sif-file-name                     INITIAL.SIF
!  initial-sif-file-device                   51
!  no-primal-objective                       NO
!  no-dual-objective                         NO
!  scale-problem                             0
!  pre-solve-problem                         YES
!  write-presolved-sif                       NO
!  presolved-sif-file-name                   PRESOLVE.SIF
!  presolved-sif-file-device                 53
!  write-scaled-sif                          NO
!  scaled-sif-file-name                      SCALED.SIF
!  scaled-sif-file-device                    58
!  solve-problem                             YES
!  print-full-solution                       NO
!  write-solution                            NO
!  solution-file-name                        WCPSOL.d
!  solution-file-device                      62
!  write-result-summary                      NO
!  result-summary-file-name                  WCPRES.d
!  result-summary-file-device                47
!  perturb-bounds-by                         0.0
! END RUNWCP SPECIFICATIONS

!  Default values for specfile-defined parameters

      INTEGER :: scale = 0
      INTEGER :: dfiledevice = 26
      INTEGER :: ifiledevice = 51
      INTEGER :: pfiledevice = 53
      INTEGER :: qfiledevice = 58
      INTEGER :: rfiledevice = 47
      INTEGER :: sfiledevice = 62
      LOGICAL :: write_problem_data   = .FALSE.
      LOGICAL :: write_initial_sif    = .FALSE.
      LOGICAL :: write_presolved_sif  = .FALSE.
      LOGICAL :: write_scaled_sif     = .FALSE.
      LOGICAL :: write_solution       = .FALSE.
      LOGICAL :: write_result_summary = .FALSE.
      CHARACTER ( LEN = 30 ) :: dfilename = 'WCP.data'
      CHARACTER ( LEN = 30 ) :: ifilename = 'INITIAL.SIF'
      CHARACTER ( LEN = 30 ) :: pfilename = 'PRESOLVE.SIF'
      CHARACTER ( LEN = 30 ) :: qfilename = 'SCALED.SIF'
      CHARACTER ( LEN = 30 ) :: rfilename = 'WCPRES.d'
      CHARACTER ( LEN = 30 ) :: sfilename = 'WCPSOL.d'
      LOGICAL :: no_primal_objective = .FALSE.
      LOGICAL :: no_dual_objective = .FALSE.
      LOGICAL :: do_presolve = .TRUE.
      LOGICAL :: do_solve = .TRUE.
      LOGICAL :: fulsol = .FALSE. 
      REAL ( KIND = wp ) :: pert_bnd = zero

!  Output file characteristics

      INTEGER, PARAMETER :: out  = 6
      INTEGER :: errout = 6
      CHARACTER ( LEN =  5 ) :: state, solv
      CHARACTER ( LEN = 30 ) :: pname, sls_solv

!  Arrays

      TYPE ( WCP_control_type ) :: WCP_control
      TYPE ( WCP_inform_type ) :: WCP_inform
      TYPE ( WCP_data_type ) :: data
      TYPE ( QPT_problem_type ) :: prob
      TYPE ( PRESOLVE_control_type ) :: PRE_control
      TYPE ( PRESOLVE_inform_type )  :: PRE_inform
      TYPE ( PRESOLVE_data_type )    :: PRE_data
      TYPE ( SCALE_trans_type ) :: SCALE_trans
      TYPE ( SCALE_data_type ) :: SCALE_data
      TYPE ( SCALE_control_type ) :: SCALE_control
      TYPE ( SCALE_inform_type ) :: SCALE_inform

!  Allocatable arrays

      CHARACTER ( LEN = 10 ), ALLOCATABLE, DIMENSION( : ) :: VNAME, CNAME
      REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: C, AY, HX
      LOGICAL, ALLOCATABLE, DIMENSION( : ) :: EQUATN, LINEAR
      INTEGER, ALLOCATABLE, DIMENSION( : ) :: IW

      CALL CPU_TIME( time ) ; CALL CLOCK_time( clock )

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

      CALL CUTEST_csetup( cutest_status, input, out, io_buffer, n, m,          &
                   prob%X, prob%X_l, prob%X_u, prob%Y_l,                       &
                   prob%C_l, prob%C_u, EQUATN, LINEAR, 0, 0, 0 )
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

      CALL CUTEST_cdimsj( cutest_status, la ) ; la = MAX( la, 1 )
      IF ( cutest_status /= 0 ) GO TO 910

!  Allocate arrays to hold the Jacobian

      ALLOCATE( prob%A%row( la ), prob%A%col( la ), prob%A%val( la ),          &
                STAT = alloc_stat )
      IF ( alloc_stat /= 0 ) THEN
        WRITE( out, 2150 ) 'A', alloc_stat ; STOP
      END IF

!  Evaluate the linear terms of the constraint functions

      CALL CUTEST_csgr( cutest_status, n, m, prob%X0, prob%Y_l, .FALSE.,       &
                        nea, la, prob%A%val, prob%A%col, prob%A%row )
      IF ( cutest_status /= 0 ) GO TO 910

!  Exclude zeros; set the linear term for the objective function

      A_ne = 0
      prob%G( : n ) = zero
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
            prob%gradient_kind = GENERAL
          END IF  
        END IF
      END DO

!  Allocate and initialize dual variables.

      ALLOCATE( prob%Z_l( n ), prob%Z_u( n ), STAT = alloc_stat )
      IF ( alloc_stat /= 0 ) THEN
        WRITE( out, 2150 ) 'Z', alloc_stat
        STOP
      END IF
      prob%Z_l( : n ) = one ; prob%Z_u( : n ) = - one
      prob%Y_l( : m ) = one ; prob%Y_u( : m ) = - one

!     WRITE( out, "( ' maximum element of A = ', ES12.4 )" )                   &
!      MAXVAL( ABS( prob%A%val( : A_ne ) ) ),                                  &

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
      DEALLOCATE( IW )
      ALLOCATE( prob%A%row( 0 ) )
      IF ( ALLOCATED( prob%H%type ) ) DEALLOCATE( prob%H%type )
      CALL SMT_put( prob%H%type, 'SPARSE_BY_ROWS', smt_stat )
      ALLOCATE( prob%H%col( 0 ), prob%H%val( 0 ), prob%H%ptr( n + 1 ) )
      prob%H%ptr = 0

      prob%new_problem_structure = .TRUE.

!  Store the problem dimensions

      prob%n = n
      prob%m = m
      IF ( ALLOCATED( prob%A%type ) ) DEALLOCATE( prob%A%type )
      CALL SMT_put( prob%A%type, 'SPARSE_BY_ROWS', smt_stat )
      prob%f = objf
        
!     WRITE( out, "( ' maximum element of A = ', ES12.4 )" )                   &
!       MAXVAL( ABS( prob%A%val( : A_ne ) ) )

!  ------------------- problem set-up complete ----------------------

      CALL CPU_TIME( times ) ; CALL CLOCK_time( clocks )

!  ------------------ Open the specfile for runwcp ----------------

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
        spec( 21 )%keyword = 'perturb-bounds-by'
        spec( 22 )%keyword = 'write-scaled-sif'
        spec( 23 )%keyword = 'scaled-sif-file-name'
        spec( 24 )%keyword = 'scaled-sif-file-device'
        spec( 25 )%keyword = 'no-primal-objective'
        spec( 26 )%keyword = 'no-dual-objective'

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
        CALL SPECFILE_assign_real( spec( 21 ), pert_bnd, errout )
        CALL SPECFILE_assign_logical( spec( 22 ), write_scaled_sif, errout )
        CALL SPECFILE_assign_string ( spec( 23 ), qfilename, errout )
        CALL SPECFILE_assign_integer( spec( 24 ), qfiledevice, errout )
        CALL SPECFILE_assign_logical( spec( 25 ), no_primal_objective, errout )
        CALL SPECFILE_assign_logical( spec( 26 ), no_dual_objective, errout )
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

        DO i = 1, m
          IF (  prob%C_l( i ) /= prob%C_u( i ) ) THEN
            IF ( prob%C_l( i ) > - infinity )                                  &
              prob%C_l( i ) = prob%C_l( i ) - pert_bnd
            IF ( prob%C_u( i ) < infinity )                                    &
              prob%C_u( i ) = prob%C_u( i ) + pert_bnd
          END IF 
        END DO
      END IF 

!  Remove the primal or dual objective if required

      IF ( no_primal_objective ) prob%G( : n ) = zero

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
 
      CALL WCP_initialize( data, WCP_control, WCP_inform )
      IF ( is_specfile )                                                       &
        CALL WCP_read_specfile( WCP_control, input_specfile )

      WCP_control%restore_problem = 2

      printo = out > 0 .AND. WCP_control%print_level > 0
      printe = out > 0 .AND. WCP_control%print_level >= 0

      CALL COPYRIGHT( out, '2002' )

      WRITE( out,  "( ' Problem: ', A10 )" ) pname 
      IF ( no_primal_objective )                                               &
        WRITE( out, "( /, ' ** objective function removed as requested ** ' )" )
      IF ( printo ) WRITE( out, 2200 ) n, m, A_ne

!  If the preprocessor is to be used, or the problem to be output,
!  allocate sufficient space

      IF ( write_initial_sif .OR. do_presolve ) THEN

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
        
!  Writes the initial SIF file, if needed

        IF ( write_initial_sif ) THEN
          CALL QPT_write_to_sif( prob, pname, ifilename, ifiledevice,          &
                                 .FALSE., .FALSE., infinity )
          IF ( .NOT. ( do_presolve .OR. do_solve ) ) STOP
        END IF
      END IF

!  If required, scale the problem

      IF ( scale < 0 ) THEN
        CALL SCALE_get( prob, - scale, SCALE_trans, SCALE_data,                &
                        SCALE_control, SCALE_inform )
        IF ( SCALE_inform%status < 0 ) THEN
          WRITE( out, "( '  ERROR return from SCALE (status =', I0, ')' )" )   &
            SCALE_inform%status
          STOP
        END IF
        CALL SCALE_apply( prob, SCALE_trans, SCALE_data,                       &
                          SCALE_control, SCALE_inform )
        IF ( SCALE_inform%status < 0 ) THEN
          WRITE( out, "( '  ERROR return from SCALE (status =', I0, ')' )" )   &
            SCALE_inform%status
          STOP
        END IF
        IF ( write_scaled_sif )                                                &
          CALL QPT_write_to_sif( prob, pname, qfilename, qfiledevice,          &
                                 .FALSE., .FALSE., infinity )
      END IF

!  Presolve

      IF ( do_presolve ) THEN

        CALL CPU_TIME( timep1 )
        
!       set the control variables

        CALL PRESOLVE_initialize( PRE_control, PRE_inform, PRE_data )
        IF ( is_specfile )                                                     &
          CALL PRESOLVE_read_specfile( input_specfile, PRE_control, PRE_inform )

        IF ( PRE_inform%status /= 0 ) STOP

!       Overide some defaults

        PRE_control%infinity   = WCP_control%infinity
        PRE_control%c_accuracy = ten * WCP_control%stop_p
        PRE_control%z_accuracy = ten * WCP_control%stop_d

!  Call the presolver

        CALL PRESOLVE_apply( prob, PRE_control, PRE_inform, PRE_data )
        IF ( PRE_inform%status < 0 ) THEN
          WRITE( out, "( '  ERROR return from PRESOLVE (exitc =', I6, ')' )" ) &
            PRE_inform%status
          STOP
        END IF
        
        CALL CPU_TIME( timep2 )
        
        A_ne = MAX( 0, prob%A%ptr( prob%m + 1 ) - 1 )
        IF ( printo ) WRITE( out, 2300 ) prob%n, prob%m, A_ne,                 &
           timep2 - timep1, PRE_inform%nbr_transforms

!  If required, write a SIF file containing the presolved problem

        IF ( write_presolved_sif ) THEN
          CALL QPT_write_to_sif( prob, pname, pfilename, pfiledevice,          &
                                 .FALSE., .FALSE., PRE_control%infinity )
        END IF
      END IF

!  Call the optimizer

      qfval = objf 

      IF ( do_solve .AND. prob%n > 0 ) THEN

!  If required, scale the problem

        IF ( scale > 0 ) THEN
          CALL SCALE_get( prob, scale, SCALE_trans, SCALE_data,                &
                          SCALE_control, SCALE_inform )
          IF ( SCALE_inform%status < 0 ) THEN
            WRITE( out, "( '  ERROR return from SCALE (status =', I0, ')' )" ) &
              SCALE_inform%status
            STOP
          END IF
          CALL SCALE_apply( prob, SCALE_trans, SCALE_data,                     &
                            SCALE_control, SCALE_inform )
          IF ( SCALE_inform%status < 0 ) THEN
            WRITE( out, "( '  ERROR return from SCALE (status =', I0, ')' )" ) &
              SCALE_inform%status
            STOP
          END IF

          IF ( write_scaled_sif )                                              &
            CALL QPT_write_to_sif( prob, pname, qfilename, qfiledevice,        &
                                   .FALSE., .FALSE., infinity )
        END IF

        CALL CPU_TIME( timeo ) ; CALL CLOCK_time( clocko )
  
!       WRITE( 33, "( 'n, m = ', 2I6, ' obj = ', ES12.4 )" )                   &
!         prob%n, prob%m, prob%f
!       WRITE( 33, "( ' g ', /, ( 5ES12.4 ) )" ) prob%G( : prob%n )
!       WRITE( 33, "( ' x_l ', /, ( 5ES12.4 ) )" ) prob%X_l( : prob%n )
!       WRITE( 33, "( ' x_u ', /, ( 5ES12.4 ) )" ) prob%X_u( : prob%n )
!       WRITE( 33, "( ' x ', /, ( 5ES12.4 ) )" ) prob%X( : prob%n )
!       WRITE( 33, "( ' y_l ', /, ( 5ES12.4 ) )" ) prob%Y_l( : prob%m )
!       WRITE( 33, "( ' y_u ', /, ( 5ES12.4 ) )" ) prob%Y_u( : prob%m )
!       WRITE( 33, "( ' z_l ', /, ( 5ES12.4 ) )" ) prob%Z_l( : prob%n )
!       WRITE( 33, "( ' z_u ', /, ( 5ES12.4 ) )" ) prob%Z_u( : prob%n )
!       WRITE( 33, "( ' c_l ', /, ( 5ES12.4 ) )" ) prob%C_l( : prob%m )
!       WRITE( 33, "( ' c_u ', /, ( 5ES12.4 ) )" ) prob%C_u( : prob%m )
!       WRITE( 33, "( ' A_ptr ', /, ( 10I6 ) )" ) prob%A%ptr( : prob%m + 1 )
!       WRITE( 33, "( ' A_col ', /, ( 10I6 ) )" )                              &
!         prob%A%col( : prob%A%ptr( prob%m + 1 ) - 1 )
!       WRITE( 33, "( ' A_val ', /, ( 5ES12.4 ) )" )                           &
!         prob%A%val( : prob%A%ptr( prob%m + 1 ) - 1 )

        solv = ' WCP'
        prob%gradient_kind = 2
        DEALLOCATE( prob%X0 )

        CALL WCP_solve( prob, data, WCP_control, WCP_inform )

        qfval = WCP_inform%obj

        CALL CPU_TIME( timet ) ; CALL CLOCK_time( clockt )
  
!  Deallocate arrays from the minimization
  
        status = WCP_inform%status ; iter = WCP_inform%iter
        nfacts = WCP_inform%nfacts ; stopr = WCP_control%stop_d
        factorization_integer = WCP_inform%factorization_integer 
        factorization_real = WCP_inform%factorization_real
        CALL WCP_terminate( data, WCP_control, WCP_inform )

!  If the problem was scaled, unscale it.

        IF ( scale > 0 ) THEN
          CALL SCALE_recover( prob, SCALE_trans, SCALE_data,                   &
                              SCALE_control, SCALE_inform )
          IF ( SCALE_inform%status < 0 ) THEN
            WRITE( out, "( '  ERROR return from SCALE (status =', I0, ')' )" ) &
              SCALE_inform%status
            STOP
          END IF
        END IF
      ELSE
        timeo  = 0.0
        timet  = 0.0
        iter   = 0
        solv   = ' NONE'
        status = 0
        stopr  = WCP_control%stop_d
        nfacts = 0
        factorization_integer = 0
        factorization_real    = 0
        qfval  = prob%f
      END IF

!  Restore from presolve

      IF ( do_presolve ) THEN
        IF ( PRE_control%print_level >= DEBUG )                                &
          CALL QPT_write_problem( out, prob )
        CALL CPU_TIME( timep3 )
        CALL PRESOLVE_restore( prob, PRE_control, PRE_inform, PRE_data )
        IF ( PRE_inform%status /= 0 .AND. printo )                             &
          WRITE( out, " ( /, ' Warning: info%status following',                &
       &  ' PRESOLVE_restore is ', I5, / ) " ) PRE_inform%status
!       IF ( PRE_inform%status /= 0 ) STOP
        qfval = prob%q
        CALL PRESOLVE_terminate( PRE_control, PRE_inform, PRE_data )
        IF ( PRE_inform%status /= 0 .AND. printo )                             &
          WRITE( out, " ( /, ' Warning: info%status following',                &
       &    ' PRESOLVE_terminate is ', I5, / ) " ) PRE_inform%status
!       IF ( PRE_inform%status /= 0 ) STOP
        IF ( .NOT. do_solve ) STOP
        CALL CPU_TIME( timep4 )
        IF ( printo ) WRITE( out, 2210 )                                       &
          timep4 - timep3, timep2 - timep1 + timep4 - timep3
      ELSE
        PRE_control%print_level = TRACE
      END IF

!  If the problem was scaled, unscale it.

      IF ( scale < 0 ) THEN
        CALL SCALE_recover( prob, SCALE_trans, SCALE_data,                     &
                            SCALE_control, SCALE_inform )
        IF ( SCALE_inform%status < 0 ) THEN
          WRITE( out, "( '  ERROR return from SCALE (status =', I0, ')' )" )   &
            SCALE_inform%status
          STOP
        END IF
      END IF

!  Compute maximum contraint residual and complementary slackness

      res_c = zero ; max_cs = zero
      DO i = 1, prob%m
        dummy = zero
        DO j = prob%A%ptr( i ), prob%A%ptr( i + 1 ) - 1
          dummy = dummy +  prob%A%val( j ) * prob%X( prob%A%col( j ) )
        END DO
        res_c = MAX( res_c, MAX( zero, prob%C_l( i ) - dummy,                  &
                                       dummy - prob%C_u( i ) ) )
        IF ( prob%C_l( i ) > - infinity ) THEN
          IF ( prob%C_u( i ) < infinity ) THEN
            max_cs = MAX( max_cs,                                              &
                 MIN( ABS( ( prob%C_l( i ) - dummy ) * prob%Y_l( i ) ),        &
                      ABS( ( prob%C_u( i ) - dummy ) * prob%Y_u( i ) ) ) ) 
          ELSE
            max_cs = MAX( max_cs,                                              &
                          ABS( ( prob%C_l( i ) - dummy ) * prob%Y_l( i ) ) )
          END IF
        ELSE IF ( prob%C_u( i ) < infinity ) THEN
           max_cs = MAX( max_cs,                                               &
                    ABS( ( prob%C_u( i ) - dummy ) * prob%Y_u( i ) ) )
        END IF
      END DO

      DO i = 1, prob%n
        dummy = prob%X( i )
        IF ( prob%X_l( i ) > - infinity ) THEN
          IF ( prob%X_u( i ) < infinity ) THEN
            max_cs = MAX( max_cs,                                              &
                 MIN( ABS( ( prob%X_l( i ) - dummy ) * prob%Z_l( i ) ),        &
                      ABS( ( prob%X_u( i ) - dummy ) * prob%Z_u( i ) ) ) )
          ELSE
            max_cs = MAX( max_cs,                                              &
                          ABS( ( prob%X_l( i ) - dummy ) * prob%Z_l( i ) ) )
          END IF
        ELSE IF ( prob%X_u( i ) < infinity ) THEN
           max_cs = MAX( max_cs,                                               &
                         ABS( ( prob%X_u( i ) - dummy ) * prob%Z_u( i ) ) )
        END IF
      END DO

!  Compute maximum KKT residual

      ALLOCATE( AY( n ), HX( n ), STAT = alloc_stat )
      AY = zero ; HX = prob%G( : n )
!     prob%G( : n ) = prob%G( : n ) - prob%Z_l( : n ) - prob%Z_u( : n )
      DO i = 1, m
        DO l = prob%A%ptr( i ), prob%A%ptr( i + 1 ) - 1
          j = prob%A%col( l )
!         prob%G( j ) = prob%G( j ) &
!           - prob%A%val( l ) * ( prob%Y_l( i ) + prob%Y_u( i ) )
          AY( j ) = AY( j ) - prob%A%val( l ) * ( prob%Y_l( i ) + prob%Y_u( i ) )
        END DO
      END DO
!     DO i = 1, n
!       WRITE(6,"( i6, 4ES12.4 )" ) i, HX( i ), prob%Z_l( i ), AY( i ),        &
!                             HX( i ) - prob%Z_l( i ) - prob%Z_u( i ) + AY( i )
!     END DO
!     WRITE(6,"( ( 5ES12.4 ) ) " ) MAXVAL( ABS( prob%Z_l + prob%Z_u ) )
!     WRITE(6,"( ' G ', /, ( 5ES12.4 ) )" ) prob%G( : n )
      res_k = MAXVAL( ABS( HX - prob%Z_l - prob%Z_u + AY ) )

!  Print details of the solution obtained

      WRITE( out, 2010 ) status
      IF ( status == 0 .OR. status == - 8 .OR. status == - 4 ) THEN
        l = 4
        IF ( fulsol ) l = n 
        IF ( do_presolve ) THEN
          IF ( PRE_control%print_level >= DEBUG ) l = n
        END IF

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
            IF ( ABS( prob%X_l( i ) - prob%X_u( i ) ) < 1.0D-10 ) THEN
              state = 'FIXED'
            ELSE IF ( prob%X( i ) - prob%X_l( i ) < - ten * stopr ) THEN
              state = 'BELOW'
            ELSE IF ( ABS( prob%X( i ) - prob%X_l( i ) ) < ten * stopr ) THEN
              state = 'LOWER'
            ELSE IF ( ABS( prob%X( i ) - prob%X_u( i ) ) < ten * stopr ) THEN
              state = 'UPPER' 
            ELSE IF ( prob%X( i ) - prob%X_u( i ) > ten * stopr ) THEN
              state = 'ABOVE'
            ELSE IF ( ABS( prob%Z_l( i ) + prob%Z_u( i ) ) < ten * stopr ) THEN
              state = 'DFREE'
            END IF
            WRITE( out, 2050 ) i, VNAME( i ), state, prob%X( i ),              &
              prob%X_l( i ), prob%X_u( i ), prob%Z_l( i ) + prob%Z_u( i )
          END DO 
        END DO 

!  Compute the number of fixed and degenerate variables.

        nfixed = 0 ; ndegen = 0 
        DO i = 1, n 
          IF ( ABS( prob%X_u( i ) - prob%X_l( i ) ) < stopr ) THEN
            nfixed = nfixed + 1 
            IF ( ABS( prob%Z_l( i ) + prob%Z_u( i )) < ten * stopr )           &
              ndegen = ndegen + 1 
          ELSE IF ( MIN( ABS( prob%X( i ) - prob%X_l( i ) ),                   &
                    ABS( prob%X( i ) - prob%X_u( i ) ) ) <=                    &
                    ten * stopr ) THEN
!           MAX( ten * stopr, ABS( prob%Z_l( i ) + prob%Z_u( i ) ) ) ) THEN
!           nfixed = nfixed + 1 
            IF ( ABS( prob%Z_l( i ) + prob%Z_u( i ) ) < ten * stopr )          &
              ndegen = ndegen + 1 
          END IF 
        END DO 

!  Print details of the constraints.

        IF ( m > 0 ) THEN 

          WRITE( out, 2040 ) 
          l = 2  ; IF ( fulsol ) l = m 
          IF ( do_presolve ) THEN
            IF ( PRE_control%print_level >= DEBUG ) l = m
          END IF
          DO j = 1, 2 
            IF ( j == 1 ) THEN 
              ir = 1 ; ic = MIN( l, m ) 
            ELSE 
              IF ( ic < m - l ) WRITE( out, 2000 ) 
              ir = MAX( ic + 1, m - ic + 1 ) ; ic = m 
            END IF 
            DO i = ir, ic 
              state = ' FREE' 
              IF ( ABS( prob%C_l( i ) - prob%C_u( i ) ) < stopr ) THEN
                state = 'EQUAL' 
              ELSE IF ( prob%C( I ) - prob%C_l( i ) < - ten * stopr ) THEN
                state = 'BELOW' 
              ELSE IF ( ABS( prob%C( I ) - prob%C_l( i ) ) < ten * stopr ) THEN
                state = 'LOWER' 
              ELSE IF ( ABS( prob%C( I ) - prob%C_u( i ) ) < ten * stopr ) THEN
                state = 'UPPER' 
              ELSE IF ( prob%C( I ) - prob%C_u( i ) > ten * stopr ) THEN
                state = 'ABOVE' 
              ELSE IF ( ABS( prob%Y_l( i ) + prob%Y_u( i )) < ten * stopr ) THEN
                state = 'DFREE'
              END IF
              WRITE( out, 2130 ) i, CNAME( i ), STATE, prob%C( i ),            &
                 prob%C_l( i ), prob%C_u( i ), prob%Y_l( i ) + prob%Y_u( i ) 
            END DO 
          END DO 

!  Compute the number of equality, fixed inequality and degenerate constraints

          mequal = 0 ; mfixed = 0 ; mdegen = 0 
          DO i = 1, m 
           IF ( ABS( prob%C_l( i ) - prob%C_u( i ) ) < stopr ) THEN
              mequal = mequal + 1
              IF ( ABS( prob%Y_l( i ) + prob%Y_u( i ) ) < stopr )              &
                mdegen = mdegen + 1 
            ELSE IF ( MIN( ABS( prob%C( i ) - prob%C_l( i ) ),                 &
                      ABS( prob%C( i ) - prob%C_u( i ) ) ) <=                  &
                 ten * stopr ) THEN
!                MAX( ten * stopr, ABS( prob%Y_l( i ) + prob%Y_u( i ) ) ) ) THEN
              mfixed = mfixed + 1
              IF ( ABS( prob%Y_l( i ) + prob%Y_u( i ) ) < stopr )              &
                mdegen = mdegen + 1 
            END IF
!           IF ( ABS( prob%C( i ) - prob%C_l( i ) ) < ten * stopr .OR.         &
!                ABS( prob%C( i ) - prob%C_u( i ) ) < ten * stopr ) THEN
!             IF ( ABS( prob%C_l( i ) - prob%C_u( i ) ) < ten * stopr ) THEN 
!                mequal = mequal + 1 
!             ELSE 
!                mfixed = mfixed + 1 
!             END IF 
!           END IF 
          END DO 
        END IF 
        WRITE( out, "( /, ' Of the ', I0, ' variable', A, ', ', I0, ' ', A,    &
       &                  ' fixed' )" )                                        &
          n, TRIM( STRING_pleural( n ) ), nfixed, TRIM( STRING_are( nfixed ) )
        WRITE( out, "( ' ', I0, ' ', A, ' implicit & ',                        &
       &                    I0, ' ', A, ' dual implicit')")                    &
          WCP_inform%x_implicit,                                               &
          TRIM( STRING_are( WCP_inform%x_implicit ) ),                         &
          WCP_inform%z_implicit,                                               &
          TRIM( STRING_are( WCP_inform%z_implicit ) )
        IF ( m > 0 ) THEN 
          WRITE( out, "( ' Of the ', I0, ' constraint', A, ', ', I0,           &
         &               ' ', A, A )" ) m, TRIM( STRING_pleural( m ) ),        &
            mequal, TRIM( STRING_are( mequal ) ),                              &
            TRIM( STRING_choice( mequal, " equations", " an equation" ) )
          WRITE( out, "( ' ', I0, ' ', A, ' implicit & ',                      &
         &                    I0, ' ', A, ' dual implicit')")                  &
            WCP_inform%c_implicit,                                             &
            TRIM( STRING_are( WCP_inform%c_implicit ) ),                       &
            WCP_inform%y_implicit,                                             &
            TRIM( STRING_are( WCP_inform%y_implicit ) )
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

          DO i = 1, n 
            state = ' FREE' 
            IF ( ABS( prob%X_l( i ) - prob%X_u( i ) ) < 1.0D-10 ) THEN
              state = 'FIXED'
            ELSE IF ( prob%X( i ) - prob%X_l( i ) < - ten * stopr ) THEN
              state = 'BELOW'
            ELSE IF ( ABS( prob%X( i ) - prob%X_l( i ) ) < ten * stopr ) THEN
              state = 'LOWER'
            ELSE IF ( ABS( prob%X( i ) - prob%X_u( i ) ) < ten * stopr ) THEN
              state = 'UPPER' 
            ELSE IF ( prob%X( i ) - prob%X_u( i ) > ten * stopr ) THEN
              state = 'ABOVE'
            ELSE IF ( ABS( prob%Z_l( i ) + prob%Z_u( i ) ) < ten * stopr ) THEN
              state = 'DFREE'
            END IF
            WRITE( sfiledevice, 2050 ) i, VNAME( i ), STATE, prob%X( i ),      &
              prob%X_l( i ), prob%X_u( i ), prob%Z_l( i ) + prob%Z_u( i )
          END DO 
  
          IF ( m > 0 ) THEN 
            WRITE( sfiledevice, 2040 ) 
            DO i = 1, m 
              state = ' FREE' 
              IF ( ABS( prob%C_l( i ) - prob%C_u( i ) ) < stopr ) THEN
                state = 'EQUAL' 
              ELSE IF ( prob%C( I ) - prob%C_l( i ) < - ten * stopr ) THEN
                state = 'BELOW' 
              ELSE IF ( ABS( prob%C( I ) - prob%C_l( i ) ) < ten * stopr ) THEN
                state = 'LOWER' 
              ELSE IF ( ABS( prob%C( I ) - prob%C_u( i ) ) < ten * stopr ) THEN
                state = 'UPPER' 
              ELSE IF ( prob%C( I ) - prob%C_u( i ) > ten * stopr ) THEN
                state = 'ABOVE' 
              ELSE IF ( ABS( prob%Y_l( i ) + prob%Y_u( i ) ) < ten * stopr )   &
                  THEN
                state = 'DFREE'
              END IF
              WRITE( sfiledevice, 2130 ) i, CNAME( i ), STATE, prob%C( i ),    &
                prob%C_l( i ), prob%C_u( i ), prob%Y_l( i ) + prob%Y_u( i )
            END DO 
          END IF 
  
          WRITE( sfiledevice, 2030 ) qfval, res_c, res_k, max_cs, iter,        &
            nfacts, factorization_integer, factorization_real 
          CLOSE( sfiledevice ) 
        END IF 
      END IF 

      sls_solv = WCP_control%SBLS_control%symmetric_linear_solver
      CALL STRING_upper_word( sls_solv )
      WRITE( out, "( /, 1X, A, ' symmetric equation solver used' )" )          &
        TRIM( sls_solv )
      WRITE( out, "( ' Typically ', I0, ', ', I0,                              &
    &                ' entries in matrix, factors' )" )                        &
        WCP_inform%SBLS_inform%SLS_inform%entries,                             &
        WCP_inform%SBLS_inform%SLS_inform%entries_in_factors
      WRITE( out, "( ' Analyse, factorize & solve CPU   times =',              &
     &  3( 1X, F8.3 ), /, ' Analyse, factorize & solve clock times =',         &
     &  3( 1X, F8.3 ) )" ) WCP_inform%time%analyse, WCP_inform%time%factorize, &
        WCP_inform%time%solve, WCP_inform%time%clock_analyse,                  &
        WCP_inform%time%clock_factorize, WCP_inform%time%clock_solve

      times = times - time ; timet = timet - timeo
      clocks = clocks - clock ; clockt = clockt - clocko
      WRITE( out, "( /, ' Total CPU, clock times = ', F8.3, ', ', F8.3 )" )    &
        times + timet, clocks + clockt 
!$    WRITE( out, "( ' number of threads = ', I0 )" ) OMP_GET_MAX_THREADS( )
      WRITE( out, 2070 ) pname 

!  Compare the variants used so far

      WRITE( out, 2080 ) solv, iter, nfacts, qfval, status, clocks, clockt,    &
                         clocks + clockt 

      IF ( write_result_summary ) THEN
        BACKSPACE( rfiledevice )
        WRITE( rfiledevice, 2190 )                                             &
           pname, n, m, WCP_inform%x_implicit, WCP_inform%z_implicit,    &
           WCP_inform%c_implicit, WCP_inform%y_implicit, iter, status, clockt
      END IF

      DEALLOCATE( VNAME, CNAME, C )
      IF ( is_specfile ) CLOSE( input_specfile )
      CALL CUTEST_cterminate( cutest_status )

      RETURN

  910 CONTINUE
      WRITE( out, "( ' CUTEst error, status = ', i0, ', stopping' )")          &
        cutest_status
      status = - 98
      STOP

!  Non-executable statements

 2000 FORMAT( '      . .          .....  ..........',                          &
              '  ..........  ..........  .......... ' ) 
 2010 FORMAT( /,' Stopping with inform%status = ', I3 ) 
 2030 FORMAT( /, ' Final objective function value  ', ES22.14, /,              &
                 ' Maximum constraint violation    ', ES22.14, /,              &
                 ' Maximum dual infeasibility      ', ES22.14, /,              &
                 ' Maximum complementary slackness ', ES22.14, //,             &
          ' Total number of iterations = ',I0,', number of factorizations = ', &
          I0, /, 1X, I0, ' integer and ', I0, ' real words required',          &
          ' for the factorization' ) 
 2040 FORMAT( /, ' Constraints : ', /, '                             ',        &
                 '        <------ Bounds ------> ', /                          &
                 '      # name       state    value   ',                       &
                 '    Lower       Upper     Multiplier ' ) 
 2050 FORMAT( I7, 1X, A10, A6, 4ES12.4 ) 
 2070 FORMAT( /, ' Problem: ', A10, //,                                        &
                 '                                 objective',                 &
                 '          < ------ time ----- > ', /,                        &
                 ' Method  iterations   factors      value  ',                 &
                 '   status setup   solve   total', /,                         &
                 ' ------  ----------   -------    ---------',                 &
                 '   ------ -----    ----   -----  ' ) 
 2080 FORMAT( A5, 2I10, 6X, ES12.4, I6, 0P, 3F8.2 ) 
 2090 FORMAT( /, ' Solution : ', /,'                              ',           &
                 '        <------ Bounds ------> ', /                          &
                 '      # name       state    value   ',                       &
                 '    Lower       Upper       Dual ' ) 
 2130 FORMAT( I7, 1X, A10, A6, 4ES12.4 ) 
 2150 FORMAT( ' Allocation error, variable ', A8, ' status = ', I6 )
 2160 FORMAT( ' IOSTAT = ', I6, ' when opening file ', A9, '. Stopping ' )
 2180 FORMAT( A10 )
 2190 FORMAT( A10, 8I7, 0P, F8.2 ) 
 2200 FORMAT( /, ' problem dimensions:  n = ', I7, ' m = ', I7,                &
              ' a_ne = ', I9 )
 2300 FORMAT( /, ' after presolve - ', /,                                      &
              ' updated dimensions:  n = ', I7, ' m = ', I7,                   &
              ' a_ne = ', I9, /,                                               &
              ' preprocessing time     =', F9.2,                               &
              '        number of transformations =', I10 )
 2210 FORMAT( /, ' postprocessing complete: time = ', F9.2,                    &
              ' total processing time =', F9.2 )
 2250 FORMAT( /, ' Problem:    ', A10, /, ' Solver :   ', A5,                  &
              /, ' Objective:', ES24.16 )

!  End of subroutine USE_WCP

     END SUBROUTINE USE_WCP

!  End of module USEWCP_double

   END MODULE GALAHAD_USEWCP_double


