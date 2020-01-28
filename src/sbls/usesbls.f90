! THIS VERSION: GALAHAD 3.3 - 27/01/2020 AT 10:30 GMT.

!-*-*-*-*-*-*-*-  G A L A H A D   U S E S B L S   M O D U L E  -*-*-*-*-*-*-*-*-

!  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
!  Principal authors: Nick Gould and Dominique Orban

!  History -
!   originally released with GALAHAD Version 2.4. January 18th 2011

!  For full documentation, see
!   http://galahad.rl.ac.uk/galahad-www/specs.html

    MODULE GALAHAD_USESBLS_double

!    -----------------------------------------------
!    | CUTEst/AMPL interface to SBLS, a method for |
!    | solving block systems of linear equations   |
!    -----------------------------------------------

      USE CUTEst_interface_double
!NOT95USE GALAHAD_CPU_time
      USE GALAHAD_CLOCK
      USE GALAHAD_QPT_double
      USE GALAHAD_SORT_double, only: SORT_reorder_by_rows
      USE GALAHAD_SBLS_double
      USE GALAHAD_SPECFILE_double
      USE GALAHAD_STRING, ONLY: STRING_upper_word
      USE GALAHAD_COPYRIGHT
      USE GALAHAD_SYMBOLS,                                                     &
            GENERAL => GALAHAD_GENERAL, ALL_ZEROS => GALAHAD_ALL_ZEROS

      IMPLICIT NONE

      PRIVATE
      PUBLIC :: USE_SBLS

    CONTAINS

!-*-*-*-*-*-*-*-*-*-   U S E _ S B L S  S U B R O U T I N E   -*-*-*-*-*-*-*-*-

     SUBROUTINE USE_SBLS( input, close_input )

!  --------------------------------------------------------------------
!
!  Solve the linear system from CUTEst
!
!     ( H  A^T ) ( x ) = ( g )
!     ( A   0  ) ( y )   ( c )
!
!  using the symmetric linear solver SBLS
!
!  --------------------------------------------------------------------

!  Dummy argument

      INTEGER, INTENT( IN ) :: input
      LOGICAL, OPTIONAL, INTENT( IN ) :: close_input

!  Parameters

      INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
      REAL ( KIND = wp ), PARAMETER :: zero = 0.0_wp
      REAL ( KIND = wp ), PARAMETER :: one = 1.0_wp
      REAL ( KIND = wp ), PARAMETER :: two = 2.0_wp
      REAL ( KIND = wp ), PARAMETER :: ten = 10.0_wp
      REAL ( KIND = wp ), PARAMETER :: infinity = ten ** 19

!     INTEGER, PARAMETER :: n_k = 100, k_k = 3, in = 28
!     REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( :, : ) :: k_val
!     CHARACTER ( len = 10 ) :: filename = 'k.val'

!  Scalars

      INTEGER :: n, m, i, j, l, ir, ic, la, lh, liw, smt_stat, ns, n_total
!     INTEGER :: np1, npm
      INTEGER :: status, alloc_stat, cutest_status, iores, neh, nea, A_ne, H_ne
      INTEGER :: n_threads = 1
      REAL :: time, timeo, times, timet
      REAL ( KIND = wp ) :: clock, clocko, clocks, clockt, clockf, clockfps
      REAL :: t1, t2, c1, c2
      REAL ( KIND = wp ) :: objf, dummy
      REAL ( KIND = wp ) :: res_c, res_k
      LOGICAL :: filexx, printo, is_specfile

!  Functions

!$    INTEGER :: OMP_GET_MAX_THREADS

!  Specfile characteristics

      INTEGER, PARAMETER :: input_specfile = 34
      INTEGER, PARAMETER :: lspec = 13
      CHARACTER ( LEN = 16 ) :: specname = 'RUNSBLS'
      TYPE ( SPECFILE_item_type ), DIMENSION( lspec ) :: spec
      CHARACTER ( LEN = 16 ) :: runspec = 'RUNSBLS.SPC'

!  The default values for SBLS could have been set as:

! BEGIN RUNSBLS SPECIFICATIONS (DEFAULT)
!  write-problem-data                                NO
!  problem-data-file-name                            SBLS.data
!  problem-data-file-device                          26
!  print-full-solution                               NO
!  write-solution                                    NO
!  solution-file-name                                SBLSSOL.d
!  solution-file-device                              62
!  write-result-summary                              NO
!  result-summary-file-name                          SBLSRES.d
!  result-summary-file-device                        47
!  least-squares-qp                                  NO
!  barrier-perturbation                              1.0
!  add-slacks                                        NO
! END RUNSBLS SPECIFICATIONS

!  Default values for specfile-defined parameters

      INTEGER :: dfiledevice = 26
      INTEGER :: sfiledevice = 62
      INTEGER :: rfiledevice = 47
      LOGICAL :: write_problem_data   = .FALSE.
      LOGICAL :: write_solution       = .FALSE.
      LOGICAL :: write_result_summary = .FALSE.
      CHARACTER ( LEN = 30 ) :: dfilename = 'SBLS.data'
      CHARACTER ( LEN = 30 ) :: sfilename = 'SBLSSOL.d'
      CHARACTER ( LEN = 30 ) :: rfilename = 'SBLSRES.d'
      LOGICAL :: fulsol = .FALSE.
      REAL ( KIND = wp ) :: barrier_pert = 0.0_wp
      LOGICAL :: least_squares_qp = .FALSE.
      LOGICAL :: add_slacks = .FALSE.

!  Output file characteristics

      INTEGER, PARAMETER :: out  = 6
      INTEGER, PARAMETER :: io_buffer = 11
      INTEGER :: errout = 6
      CHARACTER ( LEN = 10 ) :: pname

!  Arrays

      TYPE ( SMT_type ) :: C
      TYPE ( SBLS_data_type ) :: data
      TYPE ( SBLS_control_type ) :: SBLS_control
      TYPE ( SBLS_inform_type ) :: SBLS_inform
      TYPE ( QPT_problem_type ) :: prob

!  Allocatable arrays

      CHARACTER ( LEN = 10 ), ALLOCATABLE, DIMENSION( : ) :: VNAME, CNAME
      REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: AY, HX, SOL
      LOGICAL, ALLOCATABLE, DIMENSION( : ) :: EQUATN, LINEAR
      INTEGER, ALLOCATABLE, DIMENSION( : ) :: IW

!  ------------------ Open the specfile for runsbls ----------------

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
        spec( 11 )%keyword = 'barrier-perturbation'
        spec( 12 )%keyword = 'least-squares-qp'
        spec( 13 )%keyword = 'add-slacks'

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
        CALL SPECFILE_assign_real( spec( 11 ), barrier_pert, errout )
        CALL SPECFILE_assign_logical( spec( 12 ), least_squares_qp, errout )
        CALL SPECFILE_assign_logical( spec( 13 ), add_slacks, errout )
      END IF

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
      IF ( add_slacks ) THEN
        ns = COUNT( .NOT. EQUATN( : m ) ) ; la = MAX( la + ns, 1 )
      ELSE
        ns = 0 ; la = MAX( la, 1 )
      END IF

!  Allocate arrays to hold the Jacobian

      ALLOCATE( prob%A%row( la ), prob%A%col( la ), prob%A%val( la ),          &
                STAT = alloc_stat )
      IF ( alloc_stat /= 0 ) THEN
        WRITE( out, 2050 ) 'A', alloc_stat ; STOP
      END IF

      A_ne = 0
      n_total = 0
      IF ( add_slacks ) THEN
        DO i = 1, m
          IF ( .NOT. EQUATN( i ) ) THEN
            n_total = n_total + 1
            A_ne = A_ne + 1
            prob%A%row( A_ne ) = i
            prob%A%col( A_ne ) = n_total
            prob%A%val( A_ne ) = - one
          END IF
        END DO
      END IF

!  Evaluate the linear terms of the constraint functions

      CALL CUTEST_csgr( cutest_status, n, m, prob%X0, prob%Y, .FALSE.,         &
                        nea, la - ns, prob%A%val( ns + 1 : ),                  &
                        prob%A%col( ns + 1 : ), prob%A%row( ns + 1 : ) )
      IF ( cutest_status /= 0 ) GO TO 910
      DEALLOCATE( prob%X0 )

!  Exclude zeros; extract the linear term for the objective function and the
!  constraint Jacobian

      prob%G( : n ) = zero
      prob%gradient_kind = ALL_ZEROS
      DO i = ns + 1, ns + nea
        IF ( prob%A%val( i ) /= zero ) THEN
          IF ( prob%A%row( i ) > 0 ) THEN
            A_ne = A_ne + 1
            prob%A%row( A_ne ) = prob%A%row( i )
            prob%A%col( A_ne ) = prob%A%col( i ) + n_total
            prob%A%val( A_ne ) = prob%A%val( i )
          ELSE
            prob%G( prob%A%col( i ) ) = prob%A%val( i )
            prob%gradient_kind = GENERAL
          END IF
        END IF
      END DO
      n_total = n_total + n
      prob%A%n = n_total ; prob%A%m = m ; prob%A%ne = A_ne

      IF ( least_squares_qp ) THEN
        lh = n_total

!  Determine the number of nonzeros in the Hessian

      ELSE
        CALL CUTEST_cdimsh( cutest_status, lh )
        IF ( cutest_status /= 0 ) GO TO 910
        lh = MAX( lh, 1 )
      END IF

!  Allocate arrays to hold the Hessian

      IF ( barrier_pert /= 0.0_wp ) THEN
        ALLOCATE( prob%H%row( lh + n_total ), prob%H%col( lh + n ),            &
                  prob%H%val( lh + n_total ), STAT = alloc_stat )
      ELSE
        ALLOCATE( prob%H%row( lh ), prob%H%col( lh ),                          &
                  prob%H%val( lh ), STAT = alloc_stat )
      END IF
      IF ( alloc_stat /= 0 ) THEN
        WRITE( out, 2050 ) 'H', alloc_stat
        STOP
      END IF

      IF ( least_squares_qp ) THEN
        H_ne = n_total
        DO i = 1, n_total
          prob%H%row( i ) = i ; prob%H%col( i ) = i ; prob%H%val( i ) = one
        END DO

!  Evaluate the Hessian of the Lagrangian function at the initial point.

      ELSE
        CALL CUTEST_csh( cutest_status, n, m, prob%X, prob%Y,                  &
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
            prob%H%row( H_ne ) = i + ns
            prob%H%col( H_ne ) = j + ns
          ELSE
            prob%H%row( H_ne ) = j + ns
            prob%H%col( H_ne ) = i + ns
          END IF
        END DO
      END IF

!  Add barrier terms

      IF ( barrier_pert /= 0.0_wp ) THEN
        DO i = 1, n
          IF ( prob%X_l( i ) > - infinity ) THEN
            H_ne = H_ne + 1
            prob%H%row( H_ne ) = i + ns
            prob%H%col( H_ne ) = i + ns
            IF ( prob%X_u( i ) < infinity ) THEN
              prob%H%val( H_ne ) = two * barrier_pert
            ELSE
              prob%H%val( H_ne ) = barrier_pert
            END IF
          ELSE IF ( prob%X_u( i ) < infinity ) THEN
            H_ne = H_ne + 1
            prob%H%row( H_ne ) = i + ns
            prob%H%col( H_ne ) = i + ns
            prob%H%val( H_ne ) = barrier_pert
          END IF
        END DO
      END IF
      prob%H%n = n_total ; prob%H%m = n_total ; prob%H%ne = H_ne

!  Allocate and initialize dual variables.

      ALLOCATE( prob%Z( n_total ), STAT = alloc_stat )
      IF ( alloc_stat /= 0 ) THEN
        WRITE( out, 2050 ) 'Z', alloc_stat
        STOP
      END IF
      prob%Z( : n_total ) = one

      liw = MAX( m, n_total ) + 1
      ALLOCATE( prob%A%ptr( m + 1 ), prob%H%ptr( n_total + 1 ) )
      ALLOCATE( IW( liw ) )

!     WRITE( 27, "( ( 3( 2I6, ES12.4 ) ) )" )                                  &
!        ( prob%H%row( i ), prob%H%col( i ), prob%H%val( i ), i = 1, H_ne )
!     WRITE( 26, "( ' H_row ', /, ( 10I6 ) )" ) prob%H%row( : H_ne )
!     WRITE( 26, "( ' H_col ', /, ( 10I6 ) )" ) prob%H%col( : H_ne )
!     WRITE( 26, "( ' H_val ', /, ( 5ES12.4 ) )" ) prob%H%val( : H_ne )

!  Transform A to row storage format

      IF ( A_ne /= 0 ) THEN
        CALL SORT_reorder_by_rows( m, n_total, A_ne, prob%A%row, prob%A%col,   &
                                   A_ne, prob%A%val, prob%A%ptr, m + 1,        &
                                   IW, liw, out, out, i )
      ELSE
        prob%A%ptr = 1
      END IF

!  Same for H

      IF ( H_ne /= 0 ) THEN
        CALL SORT_reorder_by_rows( n_total, n_total, H_ne, prob%H%row,         &
                                   prob%H%col, H_ne, prob%H%val, prob%H%ptr,   &
                                   n_total + 1, IW, liw, out, out, i )
      ELSE
        prob%H%ptr = 1
      END IF

!  Deallocate arrays holding matrix row indices

      DEALLOCATE( prob%A%row, prob%H%row, IW )
      ALLOCATE( prob%A%row( 0 ), prob%H%row( 0 ) )
      ALLOCATE( SOL( n_total + m ), STAT = alloc_stat )

      prob%new_problem_structure = .TRUE.

!  Store the problem dimensions

      prob%n = n ; prob%m = m
      IF ( ALLOCATED( prob%H%type ) ) DEALLOCATE( prob%H%type )
      CALL SMT_put( prob%H%type, 'SPARSE_BY_ROWS', smt_stat )
      IF ( ALLOCATED( prob%A%type ) ) DEALLOCATE( prob%A%type )
      CALL SMT_put( prob%A%type, 'SPARSE_BY_ROWS', smt_stat )
      prob%f = objf ; prob%rho_g = 2 * m ; prob%rho_b = 2 * n

      CALL SMT_put( C%type, 'COORDINATE', smt_stat ) ; C%ne = 0
      ALLOCATE( C%val( C%ne ), C%row( C%ne ), C%col( C%ne ) )

!  set up the right-hand side

      SOL( : ns ) = zero
      SOL( ns + 1 : n_total ) = - prob%G( : n )
      SOL( n_total + 1 : n_total + m ) = zero

      IF ( ns > 0 ) THEN
        DEALLOCATE( prob%X, STAT = alloc_stat )
        prob%n = n_total
        ALLOCATE( prob%X( prob%n ), STAT = alloc_stat )
      END IF

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
          write( out, 2060 ) iores, dfilename
          STOP
        END IF

        WRITE( dfiledevice, "( 'n, m = ', 2I6, ' obj = ', ES12.4 )" )          &
          n_total, m, prob%f
        WRITE( dfiledevice, "( ' g ', /, ( 5ES12.4 ) )" ) prob%G( : n )
        WRITE( dfiledevice, "( ' x_l ', /, ( 5ES12.4 ) )" ) prob%X_l( : n )
        WRITE( dfiledevice, "( ' x_u ', /, ( 5ES12.4 ) )" ) prob%X_u( : n )
        WRITE( dfiledevice, "( ' c_l ', /, ( 5ES12.4 ) )" ) prob%C_l( : m )
        WRITE( dfiledevice, "( ' c_u ', /, ( 5ES12.4 ) )" ) prob%C_u( : m )
        WRITE( dfiledevice, "( ' A_ptr ', /, ( 10I6 ) )" ) prob%A%ptr( : m + 1 )
        WRITE( dfiledevice, "( ' A_col ', /, ( 10I6 ) )" ) prob%A%col( : A_ne )
        WRITE( dfiledevice, "( ' A_val ', /, ( 5ES12.4 ) )" )                  &
          prob%A%val( : A_ne )
        WRITE( dfiledevice, "( ' H_ptr ', /, ( 10I6 ) )" )                     &
          prob%H%ptr( : n_total + 1 )
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

      CALL SBLS_initialize( data, SBLS_control, SBLS_inform )
      IF ( is_specfile )                                                       &
        CALL SBLS_read_specfile( SBLS_control, input_specfile )

      printo = out > 0 .AND. SBLS_control%print_level > 0
      WRITE( out, "( /, ' problem dimensions: n = ', I0, ', m = ', I0,         &
     &       ', a_ne = ', I0, ', h_ne = ', I0 )" ) n, m, A_ne, H_ne

      IF ( printo ) CALL COPYRIGHT( out, '2011' )

!write(6,*) ' nnz - A, H ', A_ne, H_ne

!  Call the solver

      IF ( prob%n > 0 ) THEN
        CALL CPU_TIME( timeo ) ; CALL CLOCK_time( clocko )

!  =================
!  solve the problem
!  =================

        IF ( printo ) WRITE( out, " ( ' ** SBLS solver used ** ' ) " )

!  factorize

        CALL CPU_TIME( t1 ) ; CALL CLOCK_time( c1 )
        CALL SBLS_form_and_factorize( prob%n, prob%m, prob%H, prob%A, C,       &
                                      data, SBLS_control, SBLS_inform )
        CALL CPU_TIME( t2 ) ; CALL CLOCK_time( c2 )
        clockf = c2 - c1
        WRITE( 6, "( ' factorize time = ', F8.3, ' clock = ', F8.3,            &
       &  ' status = ', I0, ' nnz = ',  I0 )" ) t2 - t1, c2 - c1,              &
         SBLS_inform%status, SBLS_inform%SLS_inform%entries_in_factors
        IF ( SBLS_inform%rank_def ) WRITE( 6, "( ' * rank defficient,',        &
       &  ' nullity = ', I0 )" ) prob%m - SBLS_inform%rank
        IF ( SBLS_inform%SLS_inform%first_modified_pivot > 0 )                 &
          WRITE( 6, "( ' max mods = ', ES10.4 )" )                             &
            SBLS_inform%SLS_inform%largest_modified_pivot
        WRITE( 6, "( ' external  time = ', F8.3, ' clock = ', F8.3 )" )        &
          SBLS_inform%SLS_inform%time%analyse_external +                       &
            SBLS_inform%SLS_inform%time%factorize_external,                    &
          SBLS_inform%SLS_inform%time%clock_analyse_external +                 &
            SBLS_inform%SLS_inform%time%clock_factorize_external

        IF ( SBLS_inform%SLS_inform%num_zero /= 0 )                            &
          WRITE( 6, "( 1X, I0, ' zero diagonals ignored ' )" )                 &
            SBLS_inform%SLS_inform%num_zero

!  solve

        CALL CPU_TIME( t1 ) ; CALL CLOCK_time( c1 )
        IF ( SBLS_inform%status >= 0 ) THEN
          CALL SBLS_solve( prob%n, prob%m, prob%A, C,                          &
                           data, SBLS_control, SBLS_inform, SOL )

          prob%X( : prob%n ) = SOL( : prob%n )
          prob%Y( : prob%m ) = SOL( prob%n + 1 : prob%n + prob%m )
        END IF
        CALL CPU_TIME( t2 ) ; CALL CLOCK_time( c2 )
        clockfps = clockf + c2 - c1
        IF ( printo ) WRITE( out, " ( /, ' ** SBLS solver used ** ' ) " )

        CALL CPU_TIME( timet ) ; CALL CLOCK_time( clockt )

!  Deallocate arrays from the minimization

        status = SBLS_inform%status
        CALL SBLS_terminate( data, SBLS_control, SBLS_inform )
      ELSE
        status = 0
      END IF

      IF ( SBLS_inform%factorization == 1 ) THEN
        WRITE( out, "( /, ' Solver: ', A )" )                                  &
          TRIM( SBLS_control%definite_linear_solver )
      ELSE
        WRITE( out, "( /, ' Solver: ', A )" )                                  &
          TRIM( SBLS_control%symmetric_linear_solver )
      END IF
      WRITE( out, "(  ' Stopping with inform%status = ', I0 )" ) status

!  Compute maximum contraint residual

      IF ( status >= 0 ) THEN
        res_c = zero
        DO i = 1, m
          dummy = zero
          DO j = prob%A%ptr( i ), prob%A%ptr( i + 1 ) - 1
            dummy = dummy +  prob%A%val( j ) * prob%X( prob%A%col( j ) )
          END DO
          res_c = MAX( res_c, dummy )
        END DO

!  Compute maximum KKT residual

        ALLOCATE( AY( n_total ), HX( n_total ), STAT = alloc_stat )
        AY = zero ; HX( : ns ) = zero ; HX( ns + 1 : n_total ) = prob%G( : n )

        DO i = 1, m
          DO l = prob%A%ptr( i ), prob%A%ptr( i + 1 ) - 1
            j = prob%A%col( l )
            AY( j ) = AY( j ) + prob%A%val( l ) * prob%Y( i )
          END DO
        END DO
        DO i = 1, n_total
          DO l = prob%H%ptr( i ), prob%H%ptr( i + 1 ) - 1
            j = prob%H%col( l )
            HX( i ) = HX( i ) + prob%H%val( l ) * prob%X( j )
            IF ( j /= i )                                                      &
              HX( j ) = HX( j ) + prob%H%val( l ) * prob%X( i )
          END DO
        END DO
        res_k = MAXVAL( ABS( HX( : n_total ) + AY( : n_total ) ) )

!  Print details of the solution obtained

        IF ( status == GALAHAD_ok .OR.                                         &
             status == GALAHAD_error_cpu_limit .OR.                            &
             status == GALAHAD_error_max_iterations  .OR.                      &
             status == GALAHAD_error_tiny_step .OR.                            &
             status == GALAHAD_error_ill_conditioned ) THEN
          l = 4
          IF ( fulsol ) l = n_total

!  Print details of the primal and dual variables

          WRITE( out, 2000 )
          DO j = 1, 2
            IF ( j == 1 ) THEN
              ir = 1 ; ic = MIN( l, n_total )
            ELSE
              IF ( ic < n_total - l ) WRITE( out, 2010 )
              ir = MAX( ic + 1, n_total - ic + 1 ) ; ic = n_total
            END IF
            DO i = ir, ic
              IF ( i <= ns ) THEN
                WRITE( out, 2040 ) i, 'slack     ', prob%X( i )
              ELSE
                WRITE( out, 2040 ) i, VNAME( i - ns ), prob%X( i )
              END IF
            END DO
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
                WRITE( out, 2040 ) i, CNAME( i ), prob%C( i )
              END DO
            END DO
          END IF
          WRITE( out, 2030 ) res_c, res_k

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
              write( out, 2060 ) iores, sfilename
              STOP
            END IF

            WRITE( sfiledevice, "( /, ' Problem:    ', A10 )" ) pname
            WRITE( sfiledevice, 2000 )

            DO i = 1, ns
              WRITE( sfiledevice, 2040 ) i, 'slack     ', prob%X( i )
            END DO

            DO i = ns + 1, n_total
              WRITE( sfiledevice, 2040 ) i, VNAME( i ), prob%X( i )
            END DO

            IF ( m > 0 ) THEN
              WRITE( sfiledevice, 2020 )
              DO i = 1, m
                WRITE( sfiledevice, 2040 ) i, CNAME( i ), prob%C( i )
              END DO
            END IF

            WRITE( sfiledevice, 2030 ) res_c, res_k
            CLOSE( sfiledevice )
          END IF
        END IF
      END IF

      times = times - time ; timet = timet - timeo
      clocks = clocks - clock ; clockt = clockt - clocko
      WRITE( out, "( /, ' Total time, clock = ', F0.2, ', ', F0.2 )" )         &
        times + timet, clocks + clockt
      WRITE( out, "( ' factorization used = ', I0 )" ) SBLS_inform%factorization
!$    n_threads = OMP_GET_MAX_THREADS( )
      WRITE( out, "( ' number of threads = ', I0 )" ) n_threads
      WRITE( out, "( /, ' Problem: ', A10, //,                                 &
     &                  '          < ------ time ----- > ',                    &
     &                  '  < ----- clock ---- > ', /,                          &
     &                  '   status setup   solve   total',                     &
     &                  '   setup   solve   total', /,                         &
     &                  '   ------ -----    ----   -----',                     &
     &                  '   -----   -----   -----  ' )" ) pname

!  Compare the variants used so far

      WRITE( out, "( 1X, I6, 0P, 6F8.2 )" )                                    &
        status, times, timet, times + timet , clocks, clockt, clocks + clockt

      IF ( write_result_summary ) THEN
        BACKSPACE( rfiledevice )
!       WRITE( rfiledevice, 2190 )                                             &
!          pname, n, m, iter, qfval, status, clockt
        IF ( SBLS_inform%factorization == 1 ) THEN
          IF ( status >= 0 ) THEN
            WRITE( rfiledevice, "( A10, 2I8, A10, I11, ES9.2, 2F10.2,          &
          &                        I4, I8, 2I3 )" )                            &
              pname, m, n_total, TRIM( SBLS_control%definite_linear_solver ),  &
              SBLS_inform%SLS_inform%entries_in_factors, res_k,                &
              clockf, clockfps, status, prob%m - SBLS_inform%rank,             &
              SBLS_inform%factorization, n_threads
          ELSE
            WRITE( rfiledevice, "( A10, 2I8, A10, I11, '    -    ', 2F10.2,    &
           &                       I4, I8, 2I3 )")                             &
              pname, m, n_total, TRIM( SBLS_control%definite_linear_solver ),  &
              - SBLS_inform%SLS_inform%entries_in_factors,                     &
              - clockf, - clockfps, status, prob%m                             &
              - SBLS_inform%rank, SBLS_inform%factorization, n_threads
          END IF
        ELSE
          IF ( status >= 0 ) THEN
            WRITE( rfiledevice, "( A10, 2I8, A10, I11, ES9.2, 2F10.2,          &
          &                        I4, I8, 2I3 )" )                            &
              pname, m, n_total, TRIM( SBLS_control%symmetric_linear_solver ), &
              SBLS_inform%SLS_inform%entries_in_factors, res_k,                &
              clockf, clockfps, status, prob%m - SBLS_inform%rank,             &
              SBLS_inform%factorization, n_threads
          ELSE
            WRITE( rfiledevice, "( A10, 2I8, A10, I11, '    -    ', 2F10.2,    &
           &                       I4, I8, 2I3 )")                             &
              pname, m, n_total, TRIM( SBLS_control%symmetric_linear_solver ), &
              - SBLS_inform%SLS_inform%entries_in_factors,                     &
              - clockf, - clockfps, status, prob%m                             &
              - SBLS_inform%rank, SBLS_inform%factorization, n_threads
          END IF
        END IF
      END IF

      IF ( is_specfile ) CLOSE( input_specfile )
      DEALLOCATE( prob%X, prob%X_l, prob%X_u, prob%G, VNAME,                   &
                  prob%C_l, prob%C_u, prob%Y, prob%Z, CNAME, EQUATN,           &
                  prob%C, prob%A%row, prob%A%col, prob%A%val, prob%A%ptr,      &
                  prob%H%row, prob%H%col, prob%H%val, prob%H%ptr,              &
                  prob%A%type, prob%H%type, C%type,                            &
                  SOL, C%row, C%col, C%val, AY, HX, STAT = alloc_stat )
      CALL CUTEST_cterminate( cutest_status )
      GO TO 920
      RETURN

 910  CONTINUE
      WRITE( out, "( ' CUTEst error, status = ', i0, ', stopping' )")          &
        cutest_status
      status = - 98

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

 2000 FORMAT( /, ' Solution : ', /,                                            &
                 '      # name          value   ' )
 2010 FORMAT( '      . .           ..........' )
 2020 FORMAT( /, ' Constraints : ', /,                                         &
                 '      # name          value   ' )
 2030 FORMAT( /, ' Maximum constraint violation    ', ES22.14, /,              &
                 ' Maximum dual infeasibility      ', ES22.14 )
 2040 FORMAT( I7, 1X, A10, ES12.4 )
 2050 FORMAT( ' Allocation error, variable ', A8, ' status = ', I0 )
 2060 FORMAT( ' IOSTAT = ', I6, ' when opening file ', A9, '. Stopping ' )
 2150 FORMAT( ' Allocation error, variable ', A8, ' status = ', I6 )

!  End of subroutine USE_SBLS

     END SUBROUTINE USE_SBLS

!  End of module USESBLS_double

   END MODULE GALAHAD_USESBLS_double


