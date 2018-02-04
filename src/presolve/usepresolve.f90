! THIS VERSION: GALAHAD 3.0 - 24/10/2017 AT 15:00 GMT.

!-*-*-*-*-*-*-*-*-*-  G A L A H A D   U S E P R E S O L V E -*-*-*-*-*-*-*-*-

!  Nick Gould and Ph. L. Toint, for GALAHAD productions
!  Copyright reserved
!  April 26th 2002 as runpre
!  March 14th 2003 as usepre

    MODULE GALAHAD_USEPRESOLVE_double

!  CUTEst/AMPL interface to the GALAHAD quadratic presolver.

!-------------------------------------------------------------------------------
!  Presolve  the quadratic program
!
!     minimize     1/2 x(T) H x + g(T) x
!
!     subject to     c_l <= A x <= c_u
!                    x_l <=  x <= x_u
!
!  using PRESOLVE with the its control parameters specified in the
!  RUNPRESOLVE.SPC specfile. Optionally write the sif file corresponding to
!  the unpresolved and presolved quadratic problem(s).
!
!-------------------------------------------------------------------------------

!-------------------------------------------------------------------------------
!   U s e d   m o d u l e s   a n d   s y m b o l s
!-------------------------------------------------------------------------------

!     The CUTEst interfaces

      USE CUTEst_interface_double

!     Matrix storage schemes

      USE GALAHAD_SYMBOLS,                                                     &
          ACTIVE                => GALAHAD_ACTIVE,                             &
          DEBUG                 => GALAHAD_DEBUG,                              &
          ALL_ZEROS             => GALAHAD_ALL_ZEROS,                          &
          GENERAL               => GALAHAD_GENERAL

!     QP data type

      USE GALAHAD_QPT_double

!     Presolve itself

      USE GALAHAD_PRESOLVE_double

!     Specfile stuff

      USE GALAHAD_SPECFILE_double

!     Reordering routine

      USE GALAHAD_SORT_double, only: SORT_reorder_by_rows

!     CPU timer and copyright

!NOT95USE GALAHAD_CPU_time
      USE GALAHAD_COPYRIGHT

!-------------------------------------------------------------------------------
!   P r o g r a m    v a r i a b l e s
!-------------------------------------------------------------------------------

      IMPLICIT NONE

      PRIVATE
      PUBLIC :: USE_PRESOLVE

    CONTAINS

!-*-*-*-*-*-*-*-*-*-   U S E _ Q P A  S U B R O U T I N E   -*-*-*-*-*-*-*-*-

      SUBROUTINE USE_PRESOLVE( input )

!  Dummy argument

      INTEGER, INTENT( IN ) :: input

!  Parameters

      INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
      INTEGER, PARAMETER :: inputd = 33
      INTEGER, PARAMETER :: out =  6
      INTEGER, PARAMETER :: io_buffer = 11
      REAL ( KIND = wp ), PARAMETER :: zero     = 0.0_wp
      REAL ( KIND = wp ), PARAMETER :: one      = 1.0_wp
      REAL ( KIND = wp ), PARAMETER :: infinity = 10.0_wp ** 19

!  Scalars

      INTEGER :: n, m, la, lh, liw, i, j, l, neh, nea, alloc_stat
      INTEGER :: A_ne, H_ne, A_ne_red, H_ne_red, smt_stat, cutest_status
      INTEGER :: errout = 6
      REAL    :: time1, time2, time3, time4
      CHARACTER ( LEN = 10 ) :: pname
      LOGICAL :: is_specfile

      INTEGER :: ifiledevice = 51
      INTEGER :: pfiledevice = 53
      LOGICAL :: write_initial_sif    = .FALSE.
      LOGICAL :: write_presolved_sif  = .TRUE.
      CHARACTER ( LEN = 30 ) :: ifilename = 'INITIAL.SIF'
      CHARACTER ( LEN = 30 ) :: pfilename = 'PRESOLVE.SIF'
      CHARACTER ( LEN = 16 ) :: algo_name = 'RUNPRESOLVE'
      CHARACTER ( LEN = 16 ) :: spec = 'RUNPRESOLVE.SPC'

!  Arrays

      TYPE ( QPT_problem_type )      :: prob
      TYPE ( PRESOLVE_control_type ) :: PRESOLVE_control
      TYPE ( PRESOLVE_inform_type )  :: PRESOLVE_inform
      TYPE ( PRESOLVE_data_type )    :: PRESOLVE_data
      TYPE ( SPECFILE_item_type ), DIMENSION ( 6 ) :: specs

!  Allocatable arrays

      CHARACTER ( LEN = 10 ), ALLOCATABLE, DIMENSION( : ) :: VNAME, CNAME
      REAL     ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: X0, C
      LOGICAL, ALLOCATABLE, DIMENSION( : ) :: EQUATN, LINEAR
      INTEGER, ALLOCATABLE, DIMENSION( : ) :: IW

      CALL CPU_TIME( time1 )

!  Determine the number of variables and constraints

      CALL CUTEST_cdimen( cutest_status,  input, n, m )
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

!  Determine a feasible starting point wrt to bounds.

      prob%X( : n ) = MIN( prob%X_u( : n ),                                    &
                           MAX( prob%X_l( : n ), prob%X( : n ) ) )

!  Set X0 to zero to determine the constant terms for the problem functions

      X0 = zero

!  Evaluate the constant terms of the objective and constraint
!  functions (C)

      CALL CUTEST_cfn( cutest_status, n, m, X0, prob%f, C( : m ) )
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

      A_ne          = 0
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
         i = prob%H%row( l )
         j = prob%H%col( l )
         IF ( i < 1 .OR. i > n .OR. j < 1 .OR. j > n ) CYCLE
         H_ne = H_ne + 1
         prob%H%val( H_ne ) = prob%H%val( l )
         IF ( i >= j ) THEN
            prob%H%row( H_ne ) = i
            prob%H%col( H_ne ) = j
         ELSE
            prob%H%row( H_ne ) = j
            prob%H%col( H_ne ) = i
         END IF
      END DO

!  Allocate some pointer workspace

      liw = MAX( m, n ) + 1
      ALLOCATE( prob%A%ptr( m + 1 ), prob%H%ptr( n + 1 ) )
      ALLOCATE( IW( liw ) )

!  Transform A to row storage format

      IF ( A_ne /= 0 ) THEN
         CALL SORT_reorder_by_rows( m, n, A_ne, prob%A%row, prob%A%col, A_ne,  &
                                    prob%A%val, prob%A%ptr, m + 1, IW, liw,    &
                                    out, out, i )
      ELSE
         prob%A%ptr = 0
      END IF
      IF ( ALLOCATED( prob%A%type ) ) DEALLOCATE( prob%A%type )
      CALL SMT_put( prob%A%type, 'SPARSE_BY_ROWS', smt_stat )

!  Same for H

      IF ( H_ne /= 0 ) THEN
         CALL SORT_reorder_by_rows( n, n, H_ne, prob%H%row, prob%H%col, H_ne,  &
                                    prob%H%val, prob%H%ptr, n + 1, IW, liw,    &
                                    out, out, i )
      ELSE
         prob%H%ptr = 0
      END IF
      IF ( ALLOCATED( prob%H%type ) ) DEALLOCATE( prob%H%type )
      CALL SMT_put( prob%H%type, 'SPARSE_BY_ROWS', smt_stat )

!  Deallocate arrays holding matrix row indices

      DEALLOCATE( prob%A%row, prob%H%row )
      DEALLOCATE( IW )
      ALLOCATE( prob%A%row( 0 ), prob%H%row( 0 ) )

      prob%new_problem_structure = .TRUE.

!  Store the problem dimensions

      prob%n = n
      prob%m = m

!  Allocate sufficient space

      ALLOCATE( prob%X_status( n ), STAT = alloc_stat )
      IF ( alloc_stat /= 0 ) THEN
        IF ( out > 0 ) WRITE( out, 2150 ) 'X_status', alloc_stat
        STOP
      END IF
      prob%X_status = ACTIVE

      ALLOCATE( prob%Z( n ), prob%Z_l( n ), prob%Z_u( n ),                     &
                STAT = alloc_stat )
      IF ( alloc_stat /= 0 ) THEN
         IF ( out > 0 ) WRITE( out, 2150 ) 'Z_lu', alloc_stat
         STOP
      END IF
      prob%Z( : n )   =   zero
      prob%Z_l( : n ) = - infinity
      prob%Z_u( : n ) =   infinity

      IF ( m > 0 ) THEN
         ALLOCATE( prob%C_status( m ), STAT = alloc_stat )
         IF ( alloc_stat /= 0 ) THEN
            IF ( out > 0 ) WRITE( out, 2150 ) 'X_status', alloc_stat
            STOP
         END IF
         prob%C_status = ACTIVE

         ALLOCATE( prob%Y_l( m ), prob%Y_u( m ), STAT = alloc_stat )
         IF ( alloc_stat /= 0 ) THEN
            IF ( out > 0 ) WRITE( out, 2150 ) 'C_lu', alloc_stat
            STOP
         END IF
         prob%Y_l( : m ) = - infinity
         prob%Y_u( : m ) =   infinity
      END IF

!  ------------------- problem set-up complete ---------------------------------

      CALL CPU_TIME( time2 )

!  ------------------ Open the specfile for runpre and presolve ----------------

      INQUIRE( FILE = spec, EXIST = is_specfile )
      IF ( is_specfile ) THEN
        OPEN( inputd, FILE = spec, FORM = 'FORMATTED', STATUS='OLD')

!  ----------------- Read the runpre subset of the specfile --------------------

!   Define the keywords

        specs( 1 )%keyword = 'write-initial-sif'
        specs( 2 )%keyword = 'initial-sif-file-name'
        specs( 3 )%keyword = 'initial-sif-file-device'
        specs( 4 )%keyword = 'write-presolved-sif'
        specs( 5 )%keyword = 'presolved-sif-file-name'
        specs( 6 )%keyword = 'presolved-sif-file-device'

!   Read the specfile

        CALL SPECFILE_read( inputd, algo_name, specs, 6, errout )

!   Interpret the result

        CALL SPECFILE_assign_logical( specs( 1 ), write_initial_sif, errout )
        CALL SPECFILE_assign_string ( specs( 2 ), ifilename, errout )
        CALL SPECFILE_assign_integer( specs( 3 ), ifiledevice, errout )
        CALL SPECFILE_assign_logical( specs( 4 ), write_presolved_sif, errout )
        CALL SPECFILE_assign_string ( specs( 5 ), pfilename, errout )
        CALL SPECFILE_assign_integer( specs( 6 ), pfiledevice, errout )
      END IF

      IF ( out > 0 ) CALL COPYRIGHT( out, '2002' )

!   --------------- Write initial SIF file, if requested  ----------------------

      IF ( write_initial_sif .AND. ifiledevice > 0 ) THEN
         CALL QPT_write_to_sif( prob, pname, ifilename, ifiledevice,           &
                               .FALSE., .FALSE., infinity     )
      END IF

!  --------------------------------  Presolve ----------------------------------

!     Set the control variables to their default values

      CALL PRESOLVE_initialize( PRESOLVE_control, PRESOLVE_inform,             &
                                PRESOLVE_data )

!     Read the particular specfile for PRESOLVE

      IF ( is_specfile ) THEN
        CALL PRESOLVE_read_specfile( inputd, PRESOLVE_control, PRESOLVE_inform )
        CLOSE( inputd )
      END IF
      if ( PRESOLVE_inform%status /= 0 ) STOP

      PRESOLVE_control%infinity = infinity ! for CUTEst consistency

!     Call the presolver

      CALL CPU_TIME( time3 )
      CALL PRESOLVE_apply( prob, PRESOLVE_control, PRESOLVE_inform,            &
                           PRESOLVE_data )
      CALL CPU_TIME( time4 )

      IF ( PRESOLVE_inform%status < 0 ) THEN
         write( out, * ) '  ERROR return from PRESOLVE ( exitc =', &
              PRESOLVE_inform%status, ')'
         STOP
      END IF

      A_ne_red = MAX( 0, prob%A%ptr( prob%m + 1 ) - 1 )
      H_ne_red = MAX( 0, prob%H%ptr( prob%n + 1 ) - 1 )
      WRITE( out, 2200 ) time2 - time1, n, m, A_ne, H_ne,                      &
                         prob%n, prob%m, A_ne_red, H_ne_red,                   &
                         time4 - time3, PRESOLVE_inform%nbr_transforms

!  --------------- Write the presolved SIF file, if requested ------------------

      IF ( write_presolved_sif .AND. pfiledevice > 0 ) THEN
         CALL QPT_write_to_sif( prob, pname, pfilename, pfiledevice,           &
                                .FALSE., .FALSE., infinity )
      END IF

!  Write the presolved problem ,if DEBUG

      IF ( PRESOLVE_control%print_level >= DEBUG )                             &
        CALL QPT_write_problem( out, prob)

!  Restore the (solved) problem (for information)

!     CALL PRESOLVE_restore( prob, PRESOLVE_control, PRESOLVE_inform,          &
!                            PRESOLVE_data )
!     IF ( PRESOLVE_inform%status /= 0 ) STOP

!  Cleanup the presolving structures

      CALL PRESOLVE_terminate( PRESOLVE_control, PRESOLVE_inform,              &
                               PRESOLVE_data )
      IF ( PRESOLVE_inform%status /= 0 ) STOP

!  ---------------------- Deallocate things -----------------------------------

      DEALLOCATE( prob%X, prob%C, prob%X_l, prob%X_u, prob%G, prob%A%col,      &
                  prob%A%val, prob%H%col, prob%H%val, prob%Z, prob%A%ptr,      &
                  prob%H%ptr, prob%X_status, prob%Z_l, prob%Z_u,               &
                  VNAME, CNAME, C )
      IF ( prob%m > 0 ) DEALLOCATE( prob%C_status, prob%Y_l, prob%Y_u )
      CALL CUTEST_cterminate( cutest_status )
      RETURN

 910  CONTINUE
      WRITE( out, "( ' CUTEst error, status = ', i0, ', stopping' )" )         &
        cutest_status
      RETURN

!  Non-executable statements

 2020 FORMAT( /, ' Problem: ', A10 )
 2150 FORMAT( ' Allocation error, variable ', A8, ' status = ', I6 )
 2200 FORMAT( ' =%= setup time         =', F9.2, /,                            &
              ' =%= old dimensions:  n = ', I7, ' m = ', I7,                   &
              ' a_ne = ', I9, ' h_ne = ', I9, /,                               &
              ' =%= new dimensions:  n = ', I7, ' m = ', I7,                   &
              ' a_ne = ', I9, ' h_ne = ', I9, /,                               &
              ' =%= preprocessing time =', F9.2,                               &
              '        number of transformations =', I10 )

!  End of subroutine USE_PRESOLVE

      END SUBROUTINE USE_PRESOLVE

!  End of module USEPRESOLVE_double

    END MODULE GALAHAD_USEPRESOLVE_double

