! THIS VERSION: GALAHAD 3.3 - 27/01/2020 AT 10:30 GMT.

!-*-*-*-*-*-*-*-  G A L A H A D   U S E Q P   M O D U L E  -*-*-*-*-*-*-*-*-

!  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
!  Principal authors: Nick Gould and Dominique Orban

!  History -
!   originally released with GALAHAD Version 2.4. January 5th 2011

!  For full documentation, see
!   http://galahad.rl.ac.uk/galahad-www/specs.html

    MODULE GALAHAD_USEQP_double

!    ------------------------------------------------------
!    | CUTEst/AMPL interface to QP, a generic QP method   |
!    | that allows uniform access to other GALAHAD QP     |
!    | solvers for quadratic & least-distance programming |
!    ------------------------------------------------------

      USE CUTEst_interface_double
!NOT95USE GALAHAD_CPU_time
      USE GALAHAD_CLOCK
      USE GALAHAD_QPT_double
      USE GALAHAD_SORT_double, only: SORT_reorder_by_rows
      USE GALAHAD_QP_double
      USE GALAHAD_SPECFILE_double
      USE GALAHAD_STRING, ONLY: STRING_upper_word
      USE GALAHAD_COPYRIGHT
      USE GALAHAD_SYMBOLS,                                                     &
            GENERAL => GALAHAD_GENERAL, ALL_ZEROS => GALAHAD_ALL_ZEROS

      IMPLICIT NONE

      PRIVATE
      PUBLIC :: USE_QP

    CONTAINS

!-*-*-*-*-*-*-*-*-*-   U S E _ Q P  S U B R O U T I N E   -*-*-*-*-*-*-*-*-

     SUBROUTINE USE_QP( input )

!  --------------------------------------------------------------------
!
!  Solve the quadratic program from CUTEst
!
!     minimize     1/2 x(T) H x + g(T) x
!
!     subject to     c_l <= A x <= c_u
!                    x_l <=  x <= x_u
!
!  using the generic GALAHAD package GALAHAD_QP
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

      INTEGER :: n, m, ir, ic, la, lh, liw, iores, smt_stat
!     INTEGER :: np1, npm
      INTEGER :: i, j, l, neh, nea
      INTEGER :: status, mfixed, mdegen, nfixed, ndegen, mequal, mredun
      INTEGER :: alloc_stat, cutest_status, A_ne, H_ne, iter
      REAL :: time, timeo, times, timet
      REAL ( KIND = wp ) :: clock, clocko, clocks, clockt
      REAL ( KIND = wp ) :: objf, stopr, dummy
      REAL ( KIND = wp ) :: res_c, res_k, max_cs
      LOGICAL :: filexx, printo, is_specfile

!  Functions

!$    INTEGER :: OMP_GET_MAX_THREADS

!  Specfile characteristics

      INTEGER, PARAMETER :: input_specfile = 34
      INTEGER, PARAMETER :: lspec = 11
      CHARACTER ( LEN = 16 ) :: specname = 'RUNQP'
      TYPE ( SPECFILE_item_type ), DIMENSION( lspec ) :: spec
      CHARACTER ( LEN = 16 ) :: runspec = 'RUNQP.SPC'

!  The default values for QP could have been set as:

! BEGIN RUNQP SPECIFICATIONS (DEFAULT)
!  write-problem-data                                NO
!  problem-data-file-name                            QP.data
!  problem-data-file-device                          26
!  least-squares-qp                                  NO
!  print-full-solution                               NO
!  write-solution                                    NO
!  solution-file-name                                QPSOL.d
!  solution-file-device                              62
!  write-result-summary                              NO
!  result-summary-file-name                          QPRES.d
!  result-summary-file-device                        47
!  perturb-bounds-by                                 0.0
! END RUNQP SPECIFICATIONS

!  Default values for specfile-defined parameters

      INTEGER :: dfiledevice = 26
      INTEGER :: rfiledevice = 47
      INTEGER :: sfiledevice = 62
      LOGICAL :: write_problem_data   = .FALSE.
      LOGICAL :: write_solution       = .FALSE.
      LOGICAL :: write_result_summary = .FALSE.
      CHARACTER ( LEN = 30 ) :: dfilename = 'QP.data'
      CHARACTER ( LEN = 30 ) :: rfilename = 'QPRES.d'
      CHARACTER ( LEN = 30 ) :: sfilename = 'QPSOL.d'
      LOGICAL :: fulsol = .FALSE.
      REAL ( KIND = wp ) :: pert_bnd = zero

!  Output file characteristics

      INTEGER, PARAMETER :: out  = 6
      INTEGER, PARAMETER :: io_buffer = 11
      INTEGER :: errout = 6
      CHARACTER ( LEN =  5 ) :: state, solv
      CHARACTER ( LEN = 10 ) :: pname
      CHARACTER ( LEN = 30 ) :: sls_solv

!  Arrays

      TYPE ( QP_data_type ) :: data
      TYPE ( QP_control_type ) :: QP_control
      TYPE ( QP_inform_type ) :: QP_inform
      TYPE ( QPT_problem_type ) :: prob

!  Allocatable arrays

      CHARACTER ( LEN = 10 ), ALLOCATABLE, DIMENSION( : ) :: VNAME, CNAME
      REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: AY, HX
      LOGICAL, ALLOCATABLE, DIMENSION( : ) :: EQUATN, LINEAR
      INTEGER, ALLOCATABLE, DIMENSION( : ) :: IW, C_stat, B_stat

      CALL CPU_TIME( time ) ; CALL CLOCK_time( clock )

!  Determine the number of variables and constraints

      CALL CUTEST_cdimen( cutest_status, input, n, m )
      IF ( cutest_status /= 0 ) GO TO 910

!  Allocate suitable arrays

      ALLOCATE( prob%X( n ), prob%X_l( n ), prob%X_u( n ),                     &
                prob%G( n ), VNAME( n ), B_stat( n ), STAT = alloc_stat )
      IF ( alloc_stat /= 0 ) THEN
        WRITE( out, 2050 ) 'X', alloc_stat ; STOP
      END IF

      ALLOCATE( prob%C_l( m ), prob%C_u( m ), prob%Y( m ), CNAME( m ),         &
                EQUATN( m ), LINEAR( m ), C_stat( m ), STAT = alloc_stat )
      IF ( alloc_stat /= 0 ) THEN
        WRITE( out, 2050 ) 'C', alloc_stat ; STOP
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
        WRITE( out, 2050 ) 'X0', alloc_stat
        STOP
      END IF

      ALLOCATE( prob%C( m ), STAT = alloc_stat )
      IF ( alloc_stat /= 0 ) THEN
        WRITE( out, 2050 ) 'prob%C', alloc_stat
        STOP
      END IF

!  Determine the names of the problem, variables and constraints.

      CALL CUTEST_cnames( cutest_status, n, m, pname, VNAME, CNAME )
      IF ( cutest_status /= 0 ) GO TO 910
      WRITE( out, "( /, ' Problem: ', A10 )" ) pname

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
      DO i = 1, m
        IF ( EQUATN( i ) ) THEN
          prob%C_l( i ) = prob%C_l( i ) - prob%C( i )
          prob%C_u( i ) = prob%C_l( i )
        ELSE
          prob%C_l( i ) = prob%C_l( i ) - prob%C( i )
          prob%C_u( i ) = prob%C_u( i ) - prob%C( i )
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
        WRITE( out, 2050 ) 'A', alloc_stat ; STOP
      END IF

!  Evaluate the linear terms of the constraint functions

      CALL CUTEST_csgr( cutest_status, n, m, prob%X0, prob%Y, .FALSE.,         &
                        nea, la, prob%A%val, prob%A%col, prob%A%row )
      IF ( cutest_status /= 0 ) GO TO 910
      DEALLOCATE( prob%X0 )

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

!  Determine the number of nonzeros in the Hessian

      CALL CUTEST_cdimsh( cutest_status, lh )
      IF ( cutest_status /= 0 ) GO TO 910
      lh = MAX( lh, 1 )

!  Allocate arrays to hold the Hessian

      ALLOCATE( prob%H%row( lh ), prob%H%col( lh ), prob%H%val( lh ),          &
                STAT = alloc_stat )
      IF ( alloc_stat /= 0 ) THEN
        WRITE( out, 2050 ) 'H', alloc_stat
        STOP
      END IF

!  Evaluate the Hessian of the Lagrangian function at the initial point.

      CALL CUTEST_csh( cutest_status, n, m, prob%X, prob%Y,                    &
                       neh, lh, prob%H%val, prob%H%row, prob%H%col )
      IF ( cutest_status /= 0 ) GO TO 910
!      WRITE( out, "( ' nea = ', i8, ' la   = ', i8,                           &
!     &               ' neh  = ', i8, ' lh   = ', i8 )" ) nea, la, neh, lh

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
        WRITE( out, 2050 ) 'Z', alloc_stat
        STOP
      END IF
      prob%Z( : n ) = one

      liw = MAX( m, n ) + 1
      ALLOCATE( prob%A%ptr( m + 1 ), prob%H%ptr( n + 1 ) )
      ALLOCATE( IW( liw ) )

!     WRITE( 27, "( ( 3( 2I6, ES12.4 ) ) )" )                                  &
!        ( prob%H%row( i ), prob%H%col( i ), prob%H%val( i ), i = 1, H_ne )
!     WRITE( 26, "( ' H_row ', /, ( 10I6 ) )" ) prob%H%row( : H_ne )
!     WRITE( 26, "( ' H_col ', /, ( 10I6 ) )" ) prob%H%col( : H_ne )
!     WRITE( 26, "( ' H_val ', /, ( 5ES12.4 ) )" ) prob%H%val( : H_ne )

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

      prob%new_problem_structure = .TRUE.

!  Store the problem dimensions

      prob%n = n ; prob%m = m
      IF ( ALLOCATED( prob%H%type ) ) DEALLOCATE( prob%H%type )
      CALL SMT_put( prob%H%type, 'SPARSE_BY_ROWS', smt_stat )
      IF ( ALLOCATED( prob%A%type ) ) DEALLOCATE( prob%A%type )
      CALL SMT_put( prob%A%type, 'SPARSE_BY_ROWS', smt_stat )
      prob%f = objf

!     DO i = 1, m
!       dummy = - prob%C_l( i )
!       DO j = prob%A%ptr( i ), prob%A%ptr( i + 1 ) - 1
!         dummy = dummy +  prob%A%val( j ) * prob%X( prob%A%col( j ) )
!       END DO
!       write(6,*) i, dummy
!     END DO

!  ------------------- problem set-up complete ----------------------

      CALL CPU_TIME( times ) ; CALL CLOCK_time( clocks )

!  ------------------ Open the specfile for runqp ----------------

      INQUIRE( FILE = runspec, EXIST = is_specfile )
      IF ( is_specfile ) THEN
        OPEN( input_specfile, FILE = runspec, FORM = 'FORMATTED',              &
              STATUS = 'OLD' )

!   Define the keywords

        spec( 1 )%keyword = 'write-problem-data'
        spec( 2 )%keyword = 'problem-data-file-name'
        spec( 3 )%keyword = 'problem-data-file-device'
        spec( 4 )%keyword = 'print-full-solution'
        spec( 5 )%keyword = 'write-solution'
        spec( 6 )%keyword = 'solution-file-name'
        spec( 7 )%keyword = 'solution-file-device'
        spec( 8 )%keyword = 'write-result-summary'
        spec( 9 )%keyword = 'result-summary-file-name'
        spec( 10 )%keyword = 'result-summary-file-device'
        spec( 11 )%keyword = 'perturb-bounds-by'

!   Read the specfile

        CALL SPECFILE_read( input_specfile, specname, spec, lspec, errout )

!   Interpret the result

        CALL SPECFILE_assign_logical( spec( 1 ), write_problem_data, errout )
        CALL SPECFILE_assign_string ( spec( 2 ), dfilename, errout )
        CALL SPECFILE_assign_integer( spec( 3 ), dfiledevice, errout )
        CALL SPECFILE_assign_logical( spec( 4 ), fulsol, errout )
        CALL SPECFILE_assign_logical( spec( 5 ), write_solution, errout )
        CALL SPECFILE_assign_string ( spec( 6 ), sfilename, errout )
        CALL SPECFILE_assign_integer( spec( 7 ), sfiledevice, errout )
        CALL SPECFILE_assign_logical( spec( 8 ), write_result_summary, errout )
        CALL SPECFILE_assign_string ( spec( 9 ), rfilename, errout )
        CALL SPECFILE_assign_integer( spec( 10 ), rfiledevice, errout )
        CALL SPECFILE_assign_real( spec( 11 ), pert_bnd, errout )
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
          write( out, 2060 ) iores, dfilename
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
          write( out, 2060 ) iores, rfilename
          STOP
        END IF
        WRITE( rfiledevice, "( A10 )" ) pname
      END IF

      CALL QP_initialize( data, QP_control, QP_inform )
      IF ( is_specfile )                                                       &
        CALL QP_read_specfile( QP_control, input_specfile )

      printo = out > 0 .AND. QP_control%print_level > 0
      WRITE( out, "( /, ' problem dimensions:  n = ', I0, ', m = ', I0,        &
     &       ', a_ne = ', I0, ', h_ne = ', I0 )" ) n, m, A_ne, H_ne

      IF ( printo ) CALL COPYRIGHT( out, '2011' )

      C_stat = 0 ; B_stat = 0

!  Call the optimizer

      CALL CPU_TIME( timeo ) ; CALL CLOCK_time( clocko )
      IF ( prob%n > 0 ) THEN

!  =================
!  solve the problem
!  =================

        solv = QP_control%quadratic_programming_solver( 1 : 5 )
        CALL STRING_upper_word( solv )
        IF ( printo ) WRITE( out, " ( ' ** GALAHAD QP solver used ** ', / ) " )
        CALL QP_solve( prob, data, QP_control, QP_inform, C_stat, B_stat )

        IF ( printo ) WRITE( out, " ( /, ' ** GALAHAD QP solver used ** ' ) " )
        objf = QP_inform%obj

!  Deallocate arrays from the minimization

        status = QP_inform%status
        CALL QP_terminate( data, QP_control, QP_inform )
      ELSE
        objf = prob%f
        status = 0
        solv = ' NONE'
      END IF
      CALL CPU_TIME( timet ) ; CALL CLOCK_time( clockt )


      SELECT CASE( TRIM( QP_control%quadratic_programming_solver ) )
      CASE ( 'qpa', 'QPA' )
        stopr = QP_control%QPB_control%stop_d
        iter = QP_inform%QPA_inform%iter
        sls_solv = QP_control%QPA_control%symmetric_linear_solver
      CASE ( 'qpb', 'QPB' )
        stopr = QP_control%QPB_control%stop_d
        iter = QP_inform%QPB_inform%iter
        sls_solv = QP_control%QPB_control%SBLS_control%symmetric_linear_solver
      CASE ( 'qpc', 'QPC' )
        stopr = QP_control%QPB_control%stop_d
        iter = MAX( QP_inform%QPA_inform%iter, QP_inform%QPB_inform%iter,      &
                    QP_inform%CQP_inform%iter )
        sls_solv = QP_control%QPC_control%QPA_control%symmetric_linear_solver
      CASE ( 'cqp', 'CQP' )
        stopr = QP_control%CQP_control%stop_abs_d
        iter = QP_inform%CQP_inform%iter
        sls_solv = QP_control%CQP_control%SBLS_control%symmetric_linear_solver
      CASE DEFAULT
        stopr = QP_control%CQP_control%stop_abs_d
        iter = 0
        sls_solv = QP_control%CQP_control%SBLS_control%symmetric_linear_solver
      END SELECT
      CALL STRING_upper_word( sls_solv )

!  Compute maximum contraint residual and complementary slackness

      res_c = zero ; max_cs = zero
      DO i = 1, m
        dummy = zero
        DO j = prob%A%ptr( i ), prob%A%ptr( i + 1 ) - 1
          dummy = dummy +  prob%A%val( j ) * prob%X( prob%A%col( j ) )
        END DO
        res_c = MAX( res_c, MAX( zero, prob%C_l( i ) - dummy,                  &
                                       dummy - prob%C_u( i ) ) )
        IF ( prob%C_l( i ) > - infinity ) THEN
          IF ( prob%C_u( i ) < infinity ) THEN
            max_cs = MAX( max_cs,                                              &
                 MIN( ABS( ( prob%C_l( i ) - dummy ) * prob%Y( i ) ),          &
                      ABS( ( prob%C_u( i ) - dummy ) * prob%Y( i ) ) ) )
          ELSE
            max_cs = MAX( max_cs,                                              &
                          ABS( ( prob%C_l( i ) - dummy ) * prob%Y( i ) ) )
          END IF
        ELSE IF ( prob%C_u( i ) < infinity ) THEN
          max_cs = MAX( max_cs, ABS( ( prob%C_u( i ) - dummy ) * prob%Y( i ) ) )
        END IF
      END DO

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

      ALLOCATE( AY( n ), HX( n ), STAT = alloc_stat )
      AY = zero ; HX = prob%G( : n )
      DO i = 1, m
        DO l = prob%A%ptr( i ), prob%A%ptr( i + 1 ) - 1
          j = prob%A%col( l )
          AY( j ) = AY( j ) - prob%A%val( l ) * prob%Y( i )
        END DO
      END DO
      DO i = 1, n
        DO l = prob%H%ptr( i ), prob%H%ptr( i + 1 ) - 1
          j = prob%H%col( l )
          HX( i ) = HX( i ) + prob%H%val( l ) * prob%X( j )
          IF ( j /= i )                                                        &
            HX( j ) = HX( j ) + prob%H%val( l ) * prob%X( i )
        END DO
      END DO
      res_k = MAXVAL( ABS( HX( : n ) - prob%Z( : n ) + AY( : n ) ) )

!  Print details of the solution obtained

      WRITE( out, "( /,' Stopping with inform%status = ', I0 )" ) status
      IF ( status == GALAHAD_ok .OR.                                           &
           status == GALAHAD_error_cpu_limit .OR.                              &
           status == GALAHAD_error_max_iterations  .OR.                        &
           status == GALAHAD_error_tiny_step .OR.                              &
           status == GALAHAD_error_ill_conditioned ) THEN
        l = 4
        IF ( fulsol ) l = n

!  Print details of the primal and dual variables

        WRITE( out, 2000 )
        DO j = 1, 2
          IF ( j == 1 ) THEN
            ir = 1 ; ic = MIN( l, n )
          ELSE
            IF ( ic < n - l ) WRITE( out, 2010 )
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
            WRITE( out, 2040 ) i, VNAME( i ), state, prob%X( i ),              &
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

!  Print details of the constraints.

        IF ( m > 0 ) THEN

          WRITE( out, 2020 )
          l = 2  ; IF ( fulsol ) l = m
          DO j = 1, 2
            IF ( j == 1 ) THEN
              ir = 1 ; ic = MIN( l, m )
            ELSE
              IF ( ic < m - l ) WRITE( out, 2010 )
              ir = MAX( ic + 1, m - ic + 1 ) ; ic = m
            END IF
            DO i = ir, ic
              state = ' FREE'
              IF ( ABS( prob%C( I )   - prob%C_l( i ) ) < ten * stopr )        &
                state = 'LOWER'
              IF ( ABS( prob%C( I )   - prob%C_u( i ) ) < ten * stopr )        &
                state = 'UPPER'
              IF ( ABS( prob%C_l( i ) - prob%C_u( i ) ) <       stopr )        &
                state = 'EQUAL'
              WRITE( out, 2040 ) i, CNAME( i ), STATE, prob%C( i ),            &
                                 prob%C_l( i ), prob%C_u( i ), prob%Y( i )
            END DO
          END DO

!  Compute the number of equality, fixed inequality and degenerate constraints

          mequal = 0 ; mfixed = 0 ; mdegen = 0 ; mredun = 0
          DO i = 1, m
           IF ( ABS( prob%C_l( i ) - prob%C_u( i ) ) < stopr ) THEN
              mequal = mequal + 1
              IF ( ABS( prob%Y( i ) ) < stopr ) mredun = mredun + 1
            ELSE IF ( MIN( ABS( prob%C( i ) - prob%C_l( i ) ),                 &
                      ABS( prob%C( i ) - prob%C_u( i ) ) ) <=                  &
                 MAX( ten * stopr, ABS( prob%Y( i ) ) ) ) THEN
              mfixed = mfixed + 1
              IF ( ABS( prob%Y( i ) ) < stopr ) mdegen = mdegen + 1
            END IF
          END DO
        END IF
        WRITE( out, "( /, ' Of the ', I0, ' variables, ', I0,                  &
       &  ' are on bounds & ', I0, ' are dual degenerate' )" ) n, nfixed, ndegen
        IF ( m > 0 ) THEN
           WRITE( out, "( ' Of the ', I0, ' constraints, ', I0,                &
          &  ' are equations, & ', I0, ' are redundant' )" ) m, mequal, mredun
           IF ( m /= mequal ) WRITE( out, "( ' Of the ', I0,                   &
          &  ' inequalities, ', I0, ' are on bounds, & ', I0,                  &
          &  ' are degenerate' )" ) m - mequal, mfixed, mdegen
        END IF
        WRITE( out, 2030 ) objf, res_c, res_k, max_cs, iter

        WRITE( out, "( /, ' ** ', A, ' quadratic programming solver used **')")&
          TRIM( solv )
        WRITE( out, "( ' ** ', A, ' symmetric equation solver used **' )" )    &
          TRIM( sls_solv )

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
            write( out, 2060 ) iores, sfilename
            STOP
          END IF

          WRITE( sfiledevice, "( /, ' Problem:    ', A10, /, ' Solver :   ',   &
         &        A5, /, ' Objective:', ES24.16 )" ) pname, solv, objf
          WRITE( sfiledevice, 2000 )

          DO i = 1, n
            state = ' FREE'
            IF ( ABS( prob%X( i )   - prob%X_l( i ) ) < ten * stopr )          &
              state = 'LOWER'
            IF ( ABS( prob%X( i )   - prob%X_u( i ) ) < ten * stopr )          &
              state = 'UPPER'
            IF ( ABS( prob%X_l( I ) - prob%X_u( I ) ) < stopr )                &
              state = 'FIXED'
            WRITE( sfiledevice, 2040 ) i, VNAME( i ), STATE, prob%X( i ),      &
              prob%X_l( i ), prob%X_u( i ), prob%Z( i )
          END DO

          IF ( m > 0 ) THEN
            WRITE( sfiledevice, 2020 )
            DO i = 1, m
              state = ' FREE'
              IF ( ABS( prob%C( I ) - prob%C_l( i ) ) < ten * stopr )          &
                state = 'LOWER'
              IF ( ABS( prob%C( I ) - prob%C_u( i ) ) < ten * stopr )          &
                state = 'UPPER'
              IF ( ABS( prob%C_l( i ) - prob%C_u( i ) ) < stopr )              &
                state = 'EQUAL'
              WRITE( sfiledevice, 2040 ) i, CNAME( i ), STATE, prob%C( i ),    &
                prob%C_l( i ), prob%C_u( i ), prob%Y( i )
            END DO
          END IF

          WRITE( sfiledevice, 2030 ) objf, res_c, res_k, max_cs, iter
          CLOSE( sfiledevice )
        END IF
      END IF

      times = times - time ; timet = timet - timeo
      clocks = clocks - clock ; clockt = clockt - clocko
      WRITE( out, "( /, ' Total time, clock = ', F0.2, ', ', F0.2)" )          &
        times + timet, clocks + clockt
!$    WRITE( out, "( ' number of threads = ', I0 )" ) OMP_GET_MAX_THREADS( )
      WRITE( out, "( /, ' Problem: ', A10, //,                                 &
     &                  '                     objective',                      &
     &                  '          < ---------- time --------- > ', /,         &
     &                  ' Method  iterations    value  ',                      &
     &                  '   status setup   solve   total   clock', /,          &
     &                  ' ------  ----------   -------   ',                    &
     &                  ' ------ -----    ----   -----   -----  ' )" ) pname

!  Compare the variants used so far

      WRITE( out, "( 1X, A5, I7, 5X, ES12.4, I6, 1X, 0P, 4F8.2 )" )            &
        solv, iter, objf, status, times, timet, times + timet, clocks + clockt

      IF ( write_result_summary ) THEN
        BACKSPACE( rfiledevice )
!       WRITE( rfiledevice, 2190 )                                             &
!          pname, n, m, iter, objf, status, timet
        WRITE( rfiledevice, "( A10, ES16.8, 3ES9.1, bn, I9, F12.2, I6 )" )     &
          pname, objf, res_c, res_k, max_cs, iter, timet, status
        CLOSE( rfiledevice )
      END IF

      DEALLOCATE( VNAME, CNAME )
      IF ( is_specfile ) CLOSE( input_specfile )
      CALL CUTEST_cterminate( cutest_status )
      RETURN

 910  CONTINUE
      WRITE( out, "( ' CUTEst error, status = ', i0, ', stopping' )" )         &
        cutest_status
      status = - 98
      RETURN

!  Non-executable statements

 2000 FORMAT( /, ' Solution : ', /, '                              ',          &
                 '        <------ Bounds ------> ', /                          &
                 '      # name       state    value   ',                       &
                 '    Lower       Upper       Dual ' )
 2010 FORMAT( '      . .          .....  ..........',                          &
              '  ..........  ..........  .......... ' )
 2020 FORMAT( /, ' Constraints : ', /, '                             ',        &
                 '        <------ Bounds ------> ', /                          &
                 '      # name       state    value   ',                       &
                 '    Lower       Upper     Multiplier ' )
 2030 FORMAT( /, ' Final objective function value  ', ES22.14, /,              &
                 ' Maximum constraint violation    ', ES22.14, /,              &
                 ' Maximum dual infeasibility      ', ES22.14, /,              &
                 ' Maximum complementary slackness ', ES22.14, //,             &
                 ' Number of QP iterations = ', I0 )
 2040 FORMAT( I7, 1X, A10, A6, 4ES12.4 )
 2050 FORMAT( ' Allocation error, variable ', A8, ' status = ', I0 )
 2060 FORMAT( ' IOSTAT = ', I6, ' when opening file ', A9, '. Stopping ' )

!  End of subroutine USE_QP

     END SUBROUTINE USE_QP

!  End of module USEQP_double

   END MODULE GALAHAD_USEQP_double


