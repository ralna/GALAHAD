! THIS VERSION: GALAHAD 4.3 - 2024-01-19 AT 12:50 GMT.

#include "galahad_modules.h"
#include "cutest_routines.h"

!-*-*-*-*-*-*-*-  G A L A H A D   U S E B L L S B   M O D U L E  -*-*-*-*-*-*-*-

!  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
!  Principal authors: Nick Gould and Dominique Orban

!  History -
!   originally released with GALAHAD Version 4.3, December 28th 2023

!  For full documentation, see
!   http://galahad.rl.ac.uk/galahad-www/specs.html

    MODULE GALAHAD_USEBLLSB_precision

!     -------------------------------------------------------------------
!    | CUTEst/AMPL interface to BLLSB, an interior-point crossover       |
!    | algorithm for bound constrained linear least-squares optimization |
!     -------------------------------------------------------------------

      USE GALAHAD_KINDS_precision
!$    USE omp_lib
      USE CUTEST_INTERFACE_precision
      USE GALAHAD_CLOCK
      USE GALAHAD_QPT_precision
      USE GALAHAD_SORT_precision, ONLY: SORT_reorder_by_rows
      USE GALAHAD_NORMS_precision, ONLY: TWO_NORM
      USE GALAHAD_BLLSB_precision
      USE GALAHAD_SLS_precision
      USE GALAHAD_PRESOLVE_precision
      USE GALAHAD_SPECFILE_precision
      USE GALAHAD_STRING, ONLY: STRING_upper_word
      USE GALAHAD_COPYRIGHT
      USE GALAHAD_SYMBOLS,                                                     &
          ACTIVE    => GALAHAD_ACTIVE,                                         &
          TRACE     => GALAHAD_TRACE,                                          &
          DEBUG     => GALAHAD_DEBUG,                                          &
          GENERAL   => GALAHAD_GENERAL,                                        &
          ALL_ZEROS => GALAHAD_ALL_ZEROS
      USE GALAHAD_SCALE_precision

      IMPLICIT NONE

      PRIVATE
      PUBLIC :: USE_BLLSB

    CONTAINS

!-*-*-*-*-*-*-*-*-*-   U S E _ B L L S B  S U B R O U T I N E   -*-*-*-*-*-*-*-

     SUBROUTINE USE_BLLSB( input, close_input )

!  --------------------------------------------------------------------
!
!  Solve the bound-constrained regularized quadratic program from CUTEst
!
!     minimize     1/2 ||A_o x - b||^2_W + 1/2 sigma ||x||_2^2
!
!                    x_l <=  x <= x_u
!
!  using the GALAHAD package GALAHAD_BLLSB
!
!  --------------------------------------------------------------------

!  Dummy argument

      INTEGER ( KIND = ip_ ), INTENT( IN ) :: input
      LOGICAL, OPTIONAL, INTENT( IN ) :: close_input

!  Parameters

      REAL ( KIND = rp_ ), PARAMETER :: zero = 0.0_rp_
      REAL ( KIND = rp_ ), PARAMETER :: one = 1.0_rp_
      REAL ( KIND = rp_ ), PARAMETER :: ten = 10.0_rp_
      REAL ( KIND = rp_ ), PARAMETER :: infinity = ten ** 19

!  Scalars

      INTEGER ( KIND = ip_ ) :: n, o, m, ir, ic, iores, smt_stat
      INTEGER ( KIND = ip_ ) :: i, j, l, nnzj, n_s, nfixed, ndegen, status
      INTEGER ( KIND = ip_ ) :: alloc_stat, cutest_status, Ao_ne, iter
      REAL :: time, timeo, times, timet, timep1, timep2, timep3, timep4
      REAL ( KIND = rp_ ) :: clock, clocko, clocks, clockt, stopr, dummy
      REAL ( KIND = rp_ ) :: res_c, res_k, max_cs, max_d
      LOGICAL :: filexx, printo, printe, is_specfile
!     LOGICAL :: ldummy

!  Specfile characteristics

      INTEGER ( KIND = ip_ ), PARAMETER :: input_specfile = 34
      INTEGER ( KIND = ip_ ), PARAMETER :: lspec = 29
      CHARACTER ( LEN = 8 ) :: specname = 'RUNBLLSB'
      TYPE ( SPECFILE_item_type ), DIMENSION( lspec ) :: spec
      CHARACTER ( LEN = 12 ) :: runspec = 'RUNBLLSB.SPC'

!  The default values for BLLSB could have been set as:

! BEGIN RUNBLLSB SPECIFICATIONS (DEFAULT)
!  write-problem-data                        NO
!  problem-data-file-name                    BLLSB.data
!  problem-data-file-device                  26
!  write-initial-sif                         NO
!  initial-sif-file-name                     INITIAL.SIF
!  initial-sif-file-device                   51
!  scale-problem                             0
!  pre-solve-problem                         NO
!  write-presolved-sif                       NO
!  presolved-sif-file-name                   PRESOLVE.SIF
!  presolved-sif-file-device                 50
!  write-scaled-sif                          NO
!  scaled-sif-file-name                      SCALED.SIF
!  scaled-sif-file-device                    58
!  solve-problem                             YES
!  print-full-solution                       NO
!  write-solution                            NO
!  solution-file-name                        BLLSBSOL.d
!  solution-file-device                      62
!  write-result-summary                      NO
!  result-summary-file-name                  BLLSBRES.d
!  result-summary-file-device                47
!  perturb-bounds-by                         0.0
!  regularization-weight                     0.0
!  save-checkpoint-info                      NO
!  checkpoint-fine-name                      BLLSB.checkpoints
!  checkpoint-fine-device                    65
! END RUNBLLSB SPECIFICATIONS

!  Default values for specfile-defined parameters

      INTEGER ( KIND = ip_ ) :: scale = 0
      INTEGER ( KIND = ip_ ) :: dfiledevice = 26
      INTEGER ( KIND = ip_ ) :: ifiledevice = 51
      INTEGER ( KIND = ip_ ) :: pfiledevice = 50
      INTEGER ( KIND = ip_ ) :: qfiledevice = 58
      INTEGER ( KIND = ip_ ) :: rfiledevice = 47
      INTEGER ( KIND = ip_ ) :: sfiledevice = 62
      INTEGER ( KIND = ip_ ) :: cfiledevice = 65
      LOGICAL :: write_problem_data   = .FALSE.
      LOGICAL :: write_initial_sif    = .FALSE.
      LOGICAL :: write_presolved_sif  = .FALSE.
      LOGICAL :: write_scaled_sif     = .FALSE.
      LOGICAL :: write_solution       = .FALSE.
      LOGICAL :: write_result_summary = .FALSE.
      LOGICAL :: write_checkpoints = .FALSE.
      CHARACTER ( LEN = 30 ) :: dfilename = 'BLLSB.data'
      CHARACTER ( LEN = 30 ) :: ifilename = 'INITIAL.SIF'
      CHARACTER ( LEN = 30 ) :: pfilename = 'PRESOLVE.SIF'
      CHARACTER ( LEN = 30 ) :: qfilename = 'SCALED.SIF'
      CHARACTER ( LEN = 30 ) :: rfilename = 'BLLSBRES.d'
      CHARACTER ( LEN = 30 ) :: sfilename = 'BLLSBSOL.d'
      CHARACTER ( LEN = 30 ) :: cfilename = 'BLLSB.checkpoints'
      LOGICAL :: do_presolve = .FALSE.
      LOGICAL :: do_solve = .TRUE.
      LOGICAL :: fulsol = .FALSE.
      REAL ( KIND = rp_ ) :: pert_bnd = zero
      REAL ( KIND = rp_ ) :: regularization_weight = zero

!  Output file characteristics

      INTEGER ( KIND = ip_ ), PARAMETER :: out  = 6
      INTEGER ( KIND = ip_ ), PARAMETER :: io_buffer = 11
      INTEGER ( KIND = ip_ ) :: errout = 6
      CHARACTER ( LEN =  5 ) :: state
      CHARACTER ( LEN =  6 ) :: solv
      CHARACTER ( LEN = 10 ) :: pname
      CHARACTER ( LEN = 30 ) :: sls_solv

!  Arrays

      TYPE ( BLLSB_data_type ) :: data
      TYPE ( BLLSB_control_type ) :: control
      TYPE ( BLLSB_inform_type ) :: inform
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
      REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : ) :: X, X_l, X_u
      REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : ) :: Y, C_l, C_u, C
      LOGICAL, ALLOCATABLE, DIMENSION( : ) :: EQUATN, LINEAR

      CALL CPU_TIME( time ) ; CALL CLOCK_time( clock )

!  Determine the number of variables and constraints

      CALL CUTEST_cdimen_r( cutest_status, input, n, m )
      IF ( cutest_status /= 0 ) GO TO 910

!  allocate temporary arrays

      ALLOCATE( X( n ), X_l( n ), X_u( n ), Y( m ), C_l( m ), C_u( m ),        &
                EQUATN( m ), LINEAR( m ), STAT = alloc_stat )

      IF ( alloc_stat /= 0 ) THEN
        WRITE( out, 2150 ) 'X etc', alloc_stat ; STOP
      END IF

!  set up the data structures necessary to hold the group partially
!  separable function.

      CALL CUTEST_csetup_r( cutest_status, input, out,                         &
                            io_buffer, n, m, X, X_l, X_u, Y, C_l, C_u,         &
                            EQUATN, LINEAR, 0_ip_, 0_ip_, 0_ip_ )
      DEALLOCATE( Y, LINEAR, STAT = alloc_stat )

!  count the number of slack variables, and set problem dimensions

      n_s = m - COUNT( EQUATN )
      prob%o = m ; prob%n = n + n_s

!  Determine the names of the problem, variables and constraints.

      ALLOCATE( VNAME( prob%n ), CNAME( m ), STAT = alloc_stat )
      IF ( alloc_stat /= 0 ) THEN
        WRITE( out, 2150 ) 'VNAME etc', alloc_stat ; STOP
      END IF

      CALL CUTEST_cnames_r( cutest_status, n, m, pname, VNAME, CNAME )
      IF ( cutest_status /= 0 ) GO TO 910
      WRITE( out, "( /, ' Problem: ', A )" ) pname

!  allocate problem arrays

      ALLOCATE( prob%X( prob%n ), prob%X_l( prob%n ), prob%X_u( prob%n ),      &
                prob%B( prob%n ), C( m ), prob%Z( prob%n ), STAT = alloc_stat )
      IF ( alloc_stat /= 0 ) THEN
        WRITE( out, 2150 ) 'prob%X etc', alloc_stat ; STOP
      END IF

!  transfer data to problem

      prob%X( : n ) = X( : n )
      prob%X_l( : n ) = X_l( : n )
      prob%X_u( : n ) = X_u( : n )
      DEALLOCATE( X_l, X_u, STAT = alloc_stat )

!  determine the number of entries in the Jacobian, and set its dimensions

      CALL CUTEST_cdimsj_r( cutest_status, nnzj )
      IF ( cutest_status /= 0 ) GO TO 910
      prob%Ao%m = prob%o ; prob%Ao%n = prob%n ; prob%Ao%ne = nnzj + n_s
      CALL SMT_put( prob%Ao%type, 'COORDINATE', smt_stat )

!  allocate problem arrays

      ALLOCATE( prob%Ao%val( prob%Ao%ne ), prob%Ao%row( prob%Ao%ne ),          &
                prob%Ao%col( prob%Ao%ne ), STAT = alloc_stat )
     IF ( alloc_stat /= 0 ) THEN
        WRITE( out, 2150 ) 'prob%Ao%val etc', alloc_stat ; STOP
     END IF

!  compute the values of the constraints and Jacobian

      CALL CUTEST_ccfsg_r( cutest_status, n, m, X, C, nnzj, prob%Ao%ne,        &
                           prob%Ao%val, prob%Ao%col, prob%Ao%row, .TRUE. )
      prob%B = - C

!  deal with slack variables

      prob%Ao%ne = nnzj
      IF ( n_s > 0 ) THEN
        l = n
        DO i = 1, m
          IF ( .NOT. EQUATN( i ) ) THEN
            l = l + 1
            prob%X( l ) = zero
            prob%X_l( l ) = C_l( i )
            prob%X_u( l ) = C_u( i )
            VNAME( l ) = CNAME( i )
            prob%Ao%ne = prob%Ao%ne + 1
            prob%Ao%row( prob%Ao%ne ) = i
            prob%Ao%col( prob%Ao%ne ) = l
            prob%Ao%val( prob%Ao%ne ) = - one
          END IF
        END DO
      END IF
      DEALLOCATE( X, C, C_l, C_u, CNAME, EQUATN, STAT = alloc_stat )

!  ------------------- problem set-up complete ----------------------

      CALL CPU_TIME( times ) ; CALL CLOCK_time( clocks )

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
        spec( 25 )%keyword = 'regularization-weight'
        spec( 26 )%keyword = ''
        spec( 27 )%keyword = 'save-checkpoint-info'
        spec( 28 )%keyword = 'checkpoint-fine-name'
        spec( 29 )%keyword = 'checkpoint-fine-device'

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
        CALL SPECFILE_assign_real( spec( 25 ), regularization_weight, errout )
        CALL SPECFILE_assign_logical( spec( 27 ), write_checkpoints, errout )
        CALL SPECFILE_assign_string ( spec( 28 ), cfilename, errout )
        CALL SPECFILE_assign_integer ( spec( 29 ), cfiledevice, errout )
      END IF

!  Perturb bounds if required

      IF ( pert_bnd /= zero ) THEN
        DO i = 1, prob%n
          IF (  prob%X_l( i ) /= prob%X_u( i ) ) THEN
            IF ( prob%X_l( i ) > - infinity )                                  &
              prob%X_l( i ) = prob%X_l( i ) - pert_bnd
            IF ( prob%X_u( i ) < infinity )                                    &
              prob%X_u( i ) = prob%X_u( i ) + pert_bnd
          END IF
        END DO
      END IF

!  If required, print out the (raw) problem data

!     IF ( .TRUE. ) THEN
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

        WRITE( dfiledevice, "( 'n, o = ', 2I6, ' weight = ', ES12.4 )" )    &
          n, o, regularization_weight
!       WRITE( dfiledevice, "( ' Ao_ptr ', /, ( 10I6 ) )" )                    &
!         prob%Ao%ptr( : o + 1 )
        WRITE( dfiledevice, "( ' Ao_row ', /, ( 10I6 ) )" )                    &
          prob%Ao%row( : Ao_ne )
        WRITE( dfiledevice, "( ' Ao_col ', /, ( 10I6 ) )" )                    &
          prob%Ao%col( : Ao_ne )
        WRITE( dfiledevice, "( ' Ao_val ', /, ( 5ES12.4 ) )" )                 &
          prob%Ao%val( : Ao_ne )
        WRITE( dfiledevice, "( ' b ', /, ( 5ES12.4 ) )" ) prob%B( : m )
        WRITE( dfiledevice, "( ' x_l ', /, ( 5ES12.4 ) )" ) prob%X_l( : n )
        WRITE( dfiledevice, "( ' x_u ', /, ( 5ES12.4 ) )" ) prob%X_u( : n )
        CLOSE( dfiledevice )
      END IF

!  If required, write the checkpoint data to a file

      IF ( write_checkpoints ) THEN
        INQUIRE( FILE = cfilename, EXIST = filexx )
        IF ( filexx ) THEN
           OPEN( cfiledevice, FILE = cfilename, FORM = 'FORMATTED',            &
                 STATUS = 'OLD', POSITION = 'APPEND', IOSTAT = iores )
        ELSE
           OPEN( cfiledevice, FILE = cfilename, FORM = 'FORMATTED',            &
                 STATUS = 'NEW', IOSTAT = iores )
        END IF
        IF ( iores /= 0 ) THEN
          write( out, 2160 ) iores, cfilename
          STOP
        END IF
        WRITE( cfiledevice, 2180 ) pname
!WRITE( rfiledevice, "(A10, I8, 1X, I8 )" ) pname, n, m
!CLOSE( rfiledevice )
!STOP
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
!WRITE( rfiledevice, "(A10, I8, 1X, I8 )" ) pname, n, m
!CLOSE( rfiledevice )
!STOP
      END IF

!  Set all default values, and override defaults if requested

      CALL BLLSB_initialize( data, control, inform )

      IF ( is_specfile )                                                       &
        CALL BLLSB_read_specfile( control, input_specfile )
      IF ( scale /= 0 )                                                        &
        CALL SCALE_read_specfile( SCALE_control, input_specfile )
!     SCALE_control%print_level = control%print_level

      printo = out > 0 .AND. control%print_level > 0
      printe = out > 0 .AND. control%print_level >= 0
      WRITE( out, 2200 ) prob%n, prob%o, prob%Ao%ne

      IF ( printo ) CALL COPYRIGHT( out, '2023' )

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

!  If the preprocessor is to be used, or the problem to be output,
!  allocate sufficient space

      IF ( write_initial_sif .OR. do_presolve ) THEN

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
        prob%Z_l( : n ) = - infinity ; prob%Z_u( : n ) =   infinity

!  Writes the initial SIF file, if needed

        IF ( write_initial_sif ) THEN
          CALL QPT_write_to_sif( prob, pname, ifilename, ifiledevice,          &
                                 .FALSE., .FALSE., infinity )
          IF ( .NOT. ( do_presolve .OR. do_solve ) ) STOP
        END IF
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

        PRE_control%infinity = control%infinity
        PRE_control%c_accuracy = ten * control%stop_abs_p
        PRE_control%z_accuracy = ten * control%stop_abs_d

!  Call the presolver

        CALL PRESOLVE_apply( prob, PRE_control, PRE_inform, PRE_data )
        IF ( PRE_inform%status < 0 ) THEN
          WRITE( out, "( ' ERROR return from PRESOLVE (status =', I6, ')' )" ) &
            PRE_inform%status
          STOP
        END IF

        CALL CPU_TIME( timep2 )

        Ao_ne = MAX( 0, prob%Ao%ptr( prob%o + 1 ) - 1 )
        IF ( printo ) WRITE( out, 2300 ) prob%n, prob%o, Ao_ne,                &
           timep2 - timep1, PRE_inform%nbr_transforms

!  If required, write a SIF file containing the presolved problem

        IF ( write_presolved_sif ) THEN
          CALL QPT_write_to_sif( prob, pname, pfilename, pfiledevice,          &
                                 .FALSE., .FALSE.,                             &
                                 control%infinity )
        END IF
      END IF

!  Call the optimizer

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

!       WRITE( 6, "( ' x ', /, (5ES12.4) )" ) prob%X
!       WRITE( 6, "( ' z ', /, (5ES12.4) )" ) prob%Z

!  =================
!  solve the problem
!  =================

        solv = ' BLLSB'
        IF ( printo ) WRITE( out, " ( ' ** BLLSB solver used ** ' ) " )
        CALL BLLSB_solve( prob, data, control, inform,                         &
                          regularization_weight = regularization_weight )

        IF ( printo ) WRITE( out, " ( /, ' ** BLLSB solver used ** ' ) " )

        CALL CPU_TIME( timet ) ; CALL CLOCK_time( clockt )

!  Deallocate arrays from the minimization

        status = inform%status
        iter = inform%iter
        stopr = control%stop_abs_d
        CALL BLLSB_terminate( data, control, inform )

!  If the problem was scaled, unscale it

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
        iter  = 0
        solv   = ' NONE'
        status = 0
        stopr  = control%stop_abs_d
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

!  If the problem was scaled, unscale it

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
      max_d = MAXVAL( ABS( prob%Z( : prob%n ) ) )

      DO i = 1, prob%n
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

!  Print details of the solution obtained

      WRITE( out, 2010 ) status
      IF ( status == GALAHAD_ok .OR.                                           &
           status == GALAHAD_error_cpu_limit .OR.                              &
           status == GALAHAD_error_max_iterations  .OR.                        &
           status == GALAHAD_error_tiny_step .OR.                              &
           status == GALAHAD_error_ill_conditioned ) THEN
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
        res_k = inform%dual_infeasibility
        WRITE( out, 2030 ) inform%obj, max_d, res_c, res_k, max_cs, iter

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

          WRITE( sfiledevice, 2250 ) pname, solv, inform%obj
          WRITE( sfiledevice, 2090 )

          DO i = 1, n
            state = ' FREE'
            IF ( ABS( prob%X( i )   - prob%X_l( i ) ) < ten * stopr )          &
              state = 'LOWER'
            IF ( ABS( prob%X( i )   - prob%X_u( i ) ) < ten * stopr )          &
              state = 'UPPER'
            IF ( ABS( prob%X_l( I ) - prob%X_u( I ) ) < stopr )                &
              state = 'FIXED'
            WRITE( sfiledevice, 2050 ) i, VNAME( i ), state, prob%X( i ),      &
              prob%X_l( i ), prob%X_u( i ), prob%Z( i )
          END DO

          WRITE( sfiledevice, 2030 )                                           &
            inform%obj, max_d, res_c, res_k, max_cs, iter
          CLOSE( sfiledevice )
        END IF
      END IF

      sls_solv = inform%SLS_inform%solver
      CALL STRING_upper_word( sls_solv )
      WRITE( out, "( /, 1X, A, ' symmetric equation solver used' )" )          &
        TRIM( sls_solv )
      WRITE( out, "( ' Typically ', I0, ', ', I0,                              &
    &                ' entries in matrix, factors' )" )                        &
        inform%SLS_inform%entries, inform%SLS_inform%entries_in_factors
      WRITE( out, "( ' Analyse, factorize & solve CPU   times =',              &
     &  3( 1X, F8.3 ), /, ' Analyse, factorize & solve clock times =',         &
     &  3( 1X, F8.3 ) )" )                                                     &
        inform%time%analyse, inform%time%factorize,                            &
        inform%time%solve, inform%time%clock_analyse,                          &
        inform%time%clock_factorize, inform%time%clock_solve

      times = times - time ; timet = timet - timeo
      clocks = clocks - clock ; clockt = clockt - clocko
      WRITE( out, "( /, ' Total CPU, clock times = ', F8.3, ', ', F8.3 )" )    &
        times + timet, clocks + clockt
      WRITE( out, "( ' number of threads = ', I0 )" ) inform%threads
      WRITE( out, 2070 ) pname

!  Compare the variants used so far

      WRITE( out, 2080 ) solv, iter, inform%obj, status, clocks,               &
                         clockt, clocks + clockt

      IF ( write_checkpoints ) THEN
        BACKSPACE( cfiledevice )
        WRITE( cfiledevice, "( A10, 16( 1X, I7 ), 16( 1X, F0.2 ) )" )          &
          pname, inform%checkpointsIter( 1 : 16 ),                             &
          inform%checkpointsTime( 1 : 16 )
      END IF

      IF ( write_result_summary ) THEN
        BACKSPACE( rfiledevice )
        IF ( status >= 0 ) THEN
          WRITE( rfiledevice, "( A10, ES16.8, 3ES9.1, bn, I9, F12.2, I6 )" )   &
            pname, inform%obj, res_c, res_k, max_cs, iter, clockt, status
        ELSE
          WRITE( rfiledevice, "( A10, ES16.8, 3ES9.1, bn, I9, F12.2, I6 )" )   &
            pname, inform%obj, res_c, res_k, max_cs, - iter, - clockt, status
        END IF
        CLOSE( rfiledevice )
      END IF

      IF ( control%print_level > 5 ) THEN
        WRITE( 6, "( ' x_status')" )
        WRITE( 6, "( ( 10I8 ) )" ) prob%x_status
        WRITE( 6, "( ' x')" )
        WRITE( 6, "( ( 5ES16.8 ) )" ) prob%x
        WRITE( 6, "( ' r')" )
        WRITE( 6, "( ( 5ES16.8 ) )" ) prob%r
        WRITE( 6, "( ' z')" )
        WRITE( 6, "( ( 5ES16.8 ) )" ) prob%z
      END IF

      IF ( is_specfile ) CLOSE( input_specfile )
      DEALLOCATE( prob%X, prob%X_l, prob%X_u, prob%Z, prob%R, VNAME,           &
                  prob%Ao%row, prob%Ao%col, prob%Ao%val, prob%Ao%ptr,          &
                  prob%Ao%type, prob%X_status, STAT = alloc_stat )
      CALL CUTEST_cterminate_r( cutest_status )
      GO TO 920

 910  CONTINUE
      WRITE( out, "( ' CUTEst error, status = ', i0, ', stopping' )" )         &
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

 2000 FORMAT( '      . .          .....  ..........',                          &
              '  ..........  ..........  .......... ' )
 2010 FORMAT( /,' Stopping with inform%status = ', I0 )
 2030 FORMAT( /, ' Final objective function value  ', ES22.14, /,              &
                 ' Maximum dual variable           ', ES22.14, /,              &
                 ' Maximum constraint violation    ', ES22.14, /,              &
                 ' Maximum dual infeasibility      ', ES22.14, /,              &
                 ' Maximum complementary slackness ', ES22.14, //,             &
                 ' Number of BLLSB iterations = ', I0 )
 2050 FORMAT( I7, 1X, A10, A6, 4ES12.4 )
 2070 FORMAT( /, ' Problem: ', A, //,                                          &
                 '                     objective',                             &
                 '          < ------ time ----- > ', /,                        &
                 ' Method  iterations    value  ',                             &
                 '   status setup   solve   total', /,                         &
                 ' ------  ----------   -------   ',                           &
                 ' ------ -----    ----   -----  ' )
 2080 FORMAT( A5, I7, 6X, ES12.4, 1X, I6, 0P, 3F8.2 )
 2090 FORMAT( /, ' Solution : ', /, '                              ',          &
                 '        <------ Bounds ------> ', /                          &
                 '      # name       state    value   ',                       &
                 '    Lower       Upper       Dual ' )
 2100 FORMAT( /, ' Of the ', I0, ' variables, ', I0,                           &
              ' are on bounds & ', I0, ' are dual degenerate' )
 2150 FORMAT( ' Allocation error, variable ', A8, ' status = ', I0 )
 2160 FORMAT( ' IOSTAT = ', I6, ' when opening file ', A9, '. Stopping ' )
 2180 FORMAT( A10 )
!2190 FORMAT( A10, I7, 2I6, ES13.4, I6, 0P, F8.2 )
 2200 FORMAT( /, ' problem dimensions:  n = ', I0, ', o = ', I0,               &
                 ', ao_ne = ', I0 )
 2300 FORMAT( ' updated dimensions:  n = ', I0, ', o = ', I0,                  &
              ', ao_ne = ', I0, /, ' preprocessing time = ', F0.2,             &
              ', number of transformations = ', I0 )
 2210 FORMAT( ' postprocessing time = ', F0.2,                                 &
              ', processing time = ', F0.2 )
 2250 FORMAT( /, ' Problem:    ', A10, /, ' Solver :   ', A5,                  &
              /, ' Objective:', ES24.16 )

!  End of subroutine USE_BLLSB

     END SUBROUTINE USE_BLLSB

!  End of module USEBLLSB

   END MODULE GALAHAD_USEBLLSB_precision
