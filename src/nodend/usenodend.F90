! THIS VERSION: GALAHAD 5.2 - 2025-03-11 AT 08:45 GMT.

#include "galahad_modules.h"
#include "cutest_routines.h"

!-*-*-*-*-*-*-*-  G A L A H A D   U S E N O D E N D   M O D U L E  -*-*-*-*-*-*-

!  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
!  Principal authors: Nick Gould and Dominique Orban

!  History -
!   originally released with GALAHAD Version 5.2. March 11th 2025

!  For full documentation, see
!   http://galahad.rl.ac.uk/galahad-www/specs.html

    MODULE GALAHAD_USENODEND_precision

     USE GALAHAD_KINDS_precision

!     -----------------------------------------------------
!    | CUTEst/AMPL interface to METIS_nodend, a method for |
!    | the nested-ordering of symmetric sparse matices     |
!     -----------------------------------------------------

      USE CUTEST_INTERFACE_precision
      USE GALAHAD_CLOCK
      USE GALAHAD_COPYRIGHT
      USE GALAHAD_STRING, ONLY: STRING_leading_zero
      USE GALAHAD_SPECFILE_precision
      USE GALAHAD_NODEND_precision
      USE GALAHAD_SORT_precision, ONLY: SORT_reorder_by_cols
      USE GALAHAD_SLS_precision

      IMPLICIT NONE

      PRIVATE
      PUBLIC :: USE_NODEND

    CONTAINS

!-*-*-*-*-*-*-*-*-   U S E _ N O D E N D  S U B R O U T I N E   -*-*-*-*-*-*-

     SUBROUTINE USE_NODEND( input )

!  ---------------------------------------------------------------
!
!  Given a Hessian or Jacobian-augmented Hessian from CUTEst, find
!  a symmetric permutation so that the fill-in from Cholesky-like
!  factorization of the permuted Hessian is limited
!
!  ---------------------------------------------------------------

!  Dummy argument

      INTEGER ( KIND = ip_ ), INTENT( IN ) :: input

!  Parameters

      REAL ( KIND = rp_ ), PARAMETER :: zero = 0.0_rp_
      REAL ( KIND = rp_ ), PARAMETER :: one = 1.0_rp_
      REAL ( KIND = rp_ ), PARAMETER :: two = 2.0_rp_
      REAL ( KIND = rp_ ), PARAMETER :: ten = 10.0_rp_
      REAL ( KIND = rp_ ), PARAMETER :: infinity = ten ** 19
      REAL ( KIND = rp_ ), PARAMETER :: K22 = ten ** 6
!     LOGICAL, PARAMETER :: debug = .TRUE.
      LOGICAL, PARAMETER :: debug = .FALSE.

!  local variables

      INTEGER ( KIND = ip_ ) :: i, k, l, lh, lj, lk, lk2, iores, smt_stat
      INTEGER ( KIND = ip_ ) :: n, m, nm, nm1, nnzh, nnzj, nnzk
      INTEGER ( KIND = ip_ ) :: status, alloc_stat, cutest_status
      INTEGER ( KIND = ip_ ) :: ptype, ctype, itype, rtype, oflags
      INTEGER ( KIND = ip_ ) :: iptype, version, versions
      INTEGER ( KIND = ip_ ) :: pfactor, nseps
      REAL :: time, times, fill
      REAL ( KIND = rp_ ) :: clock, clocks
      LOGICAL :: is_specfile, filexx
      INTEGER ( KIND = ip_ ), ALLOCATABLE, DIMENSION( : ) :: K_row, K_col, K_ptr
      INTEGER ( KIND = ip_ ), ALLOCATABLE, DIMENSION( : ) :: PERM, IW
      TYPE ( NODEND_control_type ) :: control
      TYPE ( NODEND_inform_type ) :: inform

      TYPE ( SMT_type ) :: MAT
      TYPE ( SLS_data_type ) :: data
      TYPE ( SLS_control_type ) :: SLS_control
      TYPE ( SLS_inform_type ) :: SLS_inform

!  Functions

!$    INTEGER ( KIND = ip_ ) :: OMP_GET_MAX_THREADS

!  Specfile characteristics

      INTEGER ( KIND = ip_ ), PARAMETER :: input_specfile = 34
      INTEGER ( KIND = ip_ ), PARAMETER :: lspec = 11
      CHARACTER ( LEN = 16 ) :: specname = 'RUNNODEND'
      TYPE ( SPECFILE_item_type ), DIMENSION( lspec ) :: spec
      CHARACTER ( LEN = 16 ) :: runspec = 'RUNNODEND.SPC'

!  The default values for NODEND could have been set as:

! BEGIN RUNNODEND SPECIFICATIONS (DEFAULT)
!  print-permutation                                 NO
!  write-permutation                                 NO
!  permutation-file-name                             NODENDPERM.d
!  permutation-file-device                           62
!  write-result-summary                              NO
!  result-summary-file-name                          NODENDRES.d
!  result-summary-file-device                        47
!  analyse                                           NO
!  symmetric-linear-equation-solver                  ssids
!  exhaustive-tests                                  NO
!  all-versions                                      NO
! END RUNNODEND SPECIFICATIONS

!  Default values for specfile-defined parameters

      INTEGER ( KIND = ip_ ) :: pfiledevice = 62
      INTEGER ( KIND = ip_ ) :: rfiledevice = 47
      LOGICAL :: write_permutation = .FALSE.
      LOGICAL :: print_permutation = .FALSE.
      LOGICAL :: write_result_summary = .FALSE.
      LOGICAL :: analyse = .FALSE.
      LOGICAL :: exhaustive = .FALSE.
      LOGICAL :: all_versions = .FALSE.
      CHARACTER ( LEN = 30 ) :: pfilename = 'NODENDPERM.d'
      CHARACTER ( LEN = 30 ) :: rfilename = 'NODENDRES.d'
      CHARACTER ( LEN = 30 ) :: solver = "ssids" // REPEAT( ' ', 25 )

!  Output file characteristics

      INTEGER ( KIND = ip_ ), PARAMETER :: out  = 6
      INTEGER ( KIND = ip_ ), PARAMETER :: io_buffer = 11
      INTEGER ( KIND = ip_ ) :: errout = 6
      CHARACTER ( LEN = 10 ) :: pname

!  Allocatable arrays

!     CHARACTER ( LEN = 10 ), ALLOCATABLE, DIMENSION( : ) :: VNAME, CNAME
      REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : ) :: X, X_l, X_u
      REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : ) :: Y, C_l, C_u
      LOGICAL, ALLOCATABLE, DIMENSION( : ) :: EQUATN, LINEAR

      CALL COPYRIGHT( out, '2025' )

!  ------------------ Open the specfile for runnodend ----------------

      INQUIRE( FILE = runspec, EXIST = is_specfile )
      IF ( is_specfile ) THEN
        OPEN( input_specfile, FILE = runspec, FORM = 'FORMATTED',              &
              STATUS = 'OLD' )

!  define the keywords

        spec( 1 )%keyword = 'print-permutation'
        spec( 2 )%keyword = 'write-permutation'
        spec( 3 )%keyword = 'permutation-file-name'
        spec( 4 )%keyword = 'permutation-file-device'
        spec( 5 )%keyword = 'write-result-summary'
        spec( 6 )%keyword = 'result-summary-file-name'
        spec( 7 )%keyword = 'result-summary-file-device'
        spec( 8 )%keyword = 'analyse'
        spec( 9 )%keyword = 'symmetric-linear-equation-solver'
        spec( 10 )%keyword = 'exhaustive-tests'
        spec( 11 )%keyword = 'all-versions'

!  read the specfile

        CALL SPECFILE_read( input_specfile, specname, spec, lspec, errout )

!  interpret the result

        CALL SPECFILE_assign_logical( spec( 1 ), print_permutation, errout )
        CALL SPECFILE_assign_logical( spec( 2 ), write_permutation, errout )
        CALL SPECFILE_assign_string ( spec( 3 ), pfilename, errout )
        CALL SPECFILE_assign_integer( spec( 4 ), pfiledevice, errout )
        CALL SPECFILE_assign_logical( spec( 5 ), write_result_summary, errout )
        CALL SPECFILE_assign_string ( spec( 6 ), rfilename, errout )
        CALL SPECFILE_assign_integer( spec( 7 ), rfiledevice, errout )
        CALL SPECFILE_assign_logical( spec( 8 ), analyse, errout )
        CALL SPECFILE_assign_value( spec( 9 ), solver, errout )
        CALL SPECFILE_assign_logical( spec( 10 ), exhaustive, errout )
        CALL SPECFILE_assign_logical( spec( 11 ), all_versions, errout )
      END IF

!   if desired, override default control values

      CALL NODEND_read_specfile( control, input_specfile )
      IF ( .NOT. ( control%version == '5.1' .OR. control%version == '5.2' .OR. &
                   control%version == '4.0' ) ) THEN
        WRITE( out, "( ' Unknown Nodend version ', A, ' stop')") control%version
        STOP
      END IF

!  determine the number of variables and constraints

      CALL CUTEST_pname( cutest_status, input, pname )
      IF ( cutest_status /= 0 ) GO TO 910
      CALL CUTEST_cdimen_r( cutest_status, input, n, m )
      IF ( cutest_status /= 0 ) GO TO 910
      nm = n + m ; nm1 = nm + 1

!  constrained case

      IF ( m > 0 ) THEN

!  setup data structures

        ALLOCATE( X( n ), X_l( n ), X_u( n ),                                  &
                  Y( m ), C_l( m ), C_u( m ), EQUATN( m ), LINEAR( m ),        &
                  STAT = alloc_stat )
        IF ( alloc_stat /= 0 ) THEN
          WRITE( out, 2000 ) 'X', alloc_stat ; STOP
        END IF
        CALL CUTEST_csetup_r( cutest_status, input, out, io_buffer,            &
                              n, m, X, X_l, X_u, Y, C_l, C_u, EQUATN, LINEAR,  &
                              0_ip_, 0_ip_, 0_ip_ )
        DEALLOCATE( X, X_l, X_u, Y, C_l, C_u, EQUATN, LINEAR,                  &
                    STAT = alloc_stat )

!  determine the number of nonzeros in the Jacobian J and Hessian H

        CALL CUTEST_cdimsh( cutest_status, lh )
        CALL CUTEST_cdimsj( cutest_status, lj )

!  allocate space to store the row and column indices of K

        lk = lh + lj
        lk2 = 2 * lk
        ALLOCATE( K_row( lk2 ), K_col( lk2 ), K_ptr( nm1 ), IW( nm1 ),         &
                  STAT = alloc_stat )
        IF ( alloc_stat /= 0 ) THEN
          WRITE( out, 2000 ) 'K', alloc_stat ; STOP
        END IF

!  find the row and column indices

        CALL CUTEST_csgrshp( cutest_status, n, nnzj, lj,                       &
                             K_col( lh + 1 : lk ), K_row( lh + 1 : lk ),       &
                             nnzh, lh, K_row( 1 : lh ), K_col( 1 : lh ) )

!  remove gradient entries from the Jacobian

        k = lh
        DO l = lh + 1, lk
          IF ( K_row( l ) > 0 ) THEN
            k = k + 1
            K_row( k ) = K_row( l ) 
            K_col( k ) = K_col( l ) 
          END IF
        END DO
        lk = k

!  unconstrained case

      ELSE

!  setup data structures

        ALLOCATE( X( n ), X_l( n ), X_u( n ), STAT = alloc_stat )
        IF ( alloc_stat /= 0 ) THEN
          WRITE( out, 2000 ) 'X', alloc_stat ; STOP
        END IF
        CALL CUTEST_usetup_r( cutest_status, input, out, io_buffer,            &
                              n, X, X_l, X_u )
        DEALLOCATE( X, X_l, X_u, STAT = alloc_stat )

!  determine the number of nonzeros in the Hessian H

        CALL CUTEST_udimsh( cutest_status, lk )

!  allocate space to store the row and column indices of K

        lk2 = 2 * lk
        ALLOCATE( K_row( lk2 ), K_col( lk2 ), K_ptr( nm1 ), IW( nm1 ),         &
                  STAT = alloc_stat )
        IF ( alloc_stat /= 0 ) THEN
          WRITE( out, 2000 ) 'K', alloc_stat ; STOP
        END IF

!  find the row and column indices

        CALL CUTEST_ushp( cutest_status, n, nnzk, lk,                          &
                          K_row( 1 : lk ), K_col( 1 : lk ) )

      END IF

!  if required, copy the matrix structure into that required by the solver

      IF ( analyse .OR. exhaustive ) THEN
        CALL SMT_put( MAT%type, 'COORDINATE', smt_stat )
        MAT%n = nm ; MAT%ne = lk
        ALLOCATE( MAT%row( MAT%ne ), MAT%col( MAT%ne ), STAT = alloc_stat )
        MAT%row( : MAT%ne ) = K_row( 1 : lk )
        MAT%col( : MAT%ne ) = K_col( 1 : lk )
      END IF

!  remove diagonals

      k = 0
      DO l = 1, lk
        IF ( K_row( l ) /= K_col( l ) ) THEN
          k = k + 1
          K_row( k ) = K_row( l ) 
          K_col( k ) = K_col( l ) 
        END IF
      END DO

!  symmetrize the indices

      lk = k
      DO l = 1, lk
        k = k + 1
        K_row( k ) = K_col( l ) 
        K_col( k ) = K_row( l ) 
      END DO
      nnzk = k

!  reorder K to column order

      CALL SORT_reorder_by_cols( nm, nm, nnzk, K_row, K_col, lk2,              &
                                 K_ptr, nm1, IW, nm1, out, out, status )
      IF ( status > 0 ) THEN
        WRITE( out, "( ' sort error = ', I0, ' stopping' )" ) status ; STOP
      END IF

!  if desired, print the structure of the constructed input matrix

      IF ( debug ) THEN
        DO i = 1, n
          WRITE( out, "( 'col ', I0, ' row =  ',  /, ( 10I7 ) )" ) i,          &
           ( K_row( l ), l = K_ptr( i ), K_ptr( i + 1 ) - 1 )
        END DO
      END IF

!  allocate space for the permutation

      ALLOCATE( PERM( nm ), STAT = alloc_stat )
      IF ( alloc_stat /= 0 ) THEN
        WRITE( out, 2000 ) 'PERM', alloc_stat ; STOP
      END IF

!  If required, append results to a file

      IF ( write_result_summary .OR. exhaustive ) THEN
        INQUIRE( FILE = rfilename, EXIST = filexx )
        IF ( filexx ) THEN
          OPEN( rfiledevice, FILE = rfilename, FORM = 'FORMATTED',             &
                STATUS = 'OLD', POSITION = 'APPEND', IOSTAT = iores )
        ELSE
           OPEN( rfiledevice, FILE = rfilename, FORM = 'FORMATTED',            &
                 STATUS = 'NEW', IOSTAT = iores )
        END IF
        IF ( iores /= 0 ) THEN
          write( out, 2010 ) iores, rfilename
          STOP
        ELSE 
          IF ( analyse ) THEN
            IF ( .NOT. filexx ) WRITE( rfiledevice,                            &
                 & "( '--------------------------------------------',          &
                 &    '----------------------------------------', /,           &
                 &    'name           dim ver  st     clock  ct  rt',          &
                 &    '    of  ns  ni  nt nc mi co  uf  dr fill', /,           &
                 &    '                                    pt  ip  ',          &
                 &    'of                             no', /,                  &
                 &    '--------------------------------------------',          &
                 &    '----------------------------------------' )" ) 
          ELSE
            IF ( .NOT. filexx ) WRITE( rfiledevice,                            &
                 & "( '--------------------------------------------',          &
                 &    '-----------------------------------', /,                &
                 &    'name           dim ver  st     clock  ct  rt',          &
                 &    '    of  ns  ni  nt nc mi co  uf  dr', /,                &
                 &    '                                    pt  ip  ',          &
                 &    'of                             no', /,                  &
                 &    '--------------------------------------------',          &
                 &    '-----------------------------------' )" ) 
          END IF
        END IF
        IF ( .NOT. exhaustive ) WRITE( rfiledevice, "( A10 )" ) pname
      END IF

!  ---------------------------------------------------------------------
!  conduct exhaustive tests, running through all of the relevant options
!  ---------------------------------------------------------------------

      IF ( exhaustive ) THEN

!  tests for version 4

        IF ( control%version == '4.0' ) THEN
          DO ptype = 0, 1
           control%metis4_ptype = ptype
           DO ctype = 1, 4
            control%metis4_ctype = ctype
            DO itype = 1, 2
             control%metis4_itype = itype
             DO rtype = 1, 2
              control%metis4_rtype = rtype
              DO oflags = 0, 3
               control%metis4_oflags = oflags
!              DO pfactor = 1, 3
               DO pfactor = 1, 1
                IF ( pfactor == 1 ) THEN
                 control%metis4_pfactor = 0
                ELSE IF ( pfactor == 2 ) THEN
                 control%metis4_pfactor = 1
                ELSE
                 control%metis4_pfactor = 10
                END IF
                DO nseps = 1, 2
                 IF ( nseps == 1 ) THEN
                  control%metis4_nseps = 1
                 ELSE
                  control%metis4_nseps = 2
                 END IF

!  compute the ordering

write(6,"( 7I5 )" ) ptype, ctype, itype, rtype, oflags, pfactor, nseps

                 CALL CPU_TIME( times ) ; CALL CLOCK_time( clocks )
                 CALL NODEND_order_adjacency( nm, K_ptr, K_row, PERM,          &
                                              control, inform )
!write(6,*) ' inform status = ', inform%status
                 CALL CPU_TIME( time ) ; CALL CLOCK_time( clock )
                 clock = clock - clocks

!   if desired, compute the fill in

                 CALL SLS_initialize( solver, data, SLS_control, SLS_inform )
                 CALL SLS_analyse( MAT, data, SLS_control, SLS_inform,         &
                                   PERM = PERM )
                 CALL SLS_terminate( data, SLS_control, SLS_inform )
                 fill = REAL( SLS_inform%entries_in_factors ) / REAL( MAT%ne )
                 WRITE( out, "( A, ' order time = ', A, F0.2,  1X, A,          &
                &     ' analyse status = ', I0, /, ' size(matrix,factor,',     &
                &     '%fill) = ', I0, ', ', I0, ', ', F0.2 )")                &
                   TRIM( pname ), TRIM( STRING_leading_zero( clock ) ), clock, &
                   TRIM( SLS_inform%solver ), SLS_inform%status,               &
                   MAT%ne,SLS_inform%entries_in_factors, fill

!   write a summary to a file

                 IF ( inform%status < 0 ) clock = - clock
                 WRITE( rfiledevice, "( A10, I8, 1X, A3, I4, F10.2, 5I2,       &
                &                       2I4, 1X, F0.2 )" )                     &
                   pname, nm, control%version, inform%status, clock,           &
                   control%metis4_ptype, control%metis4_ctype,                 &
                   control%metis4_itype, control%metis4_rtype,                 &
                   control%metis4_oflags, control%metis4_pfactor,              &
                   control%metis4_nseps, fill
                END DO
               END DO
              END DO
             END DO
            END DO
           END DO
          END DO

!  tests for version 5

        ELSE

          DO ptype = 0, 1
           control%metis5_ptype = ptype
           DO ctype = 1, 4
            control%metis5_ctype = ctype
            DO iptype = 2, 3
             control%metis5_iptype = iptype
             DO rtype = 1, 2
              control%metis5_rtype = rtype
              DO oflags = 0, 3
                IF ( oflags == 0 ) THEN
                  control%metis5_compress = 0
                  control%metis5_ccorder = 0
                ELSE IF ( oflags == 1 ) THEN
                  control%metis5_compress = 1
                  control%metis5_ccorder = 0
                ELSE IF ( oflags == 2 ) THEN
                  control%metis5_compress = 0
                  control%metis5_ccorder = 1
                ELSE
                  control%metis5_compress = 1
                  control%metis5_ccorder = 1
                END IF
!              DO pfactor = 1, 3
               DO pfactor = 1, 1
                IF ( pfactor == 1 ) THEN
                 control%metis5_pfactor = 0
                ELSE IF ( pfactor == 2 ) THEN
                 control%metis5_pfactor = 1
                ELSE
                 control%metis5_pfactor = 10
                END IF
                DO nseps = 1, 2
                 IF ( nseps == 1 ) THEN
                  control%metis5_nseps = 1
                 ELSE
                  control%metis5_nseps = 2
                 END IF

!  compute the ordering

write(6,"( 7I5 )" ) ptype, ctype, iptype, rtype, oflags, pfactor, nseps

                 CALL CPU_TIME( times ) ; CALL CLOCK_time( clocks )
                 CALL NODEND_order_adjacency( nm, K_ptr, K_row, PERM,          &
                                              control, inform )
!write(6,*) ' inform status = ', inform%status
                 CALL CPU_TIME( time ) ; CALL CLOCK_time( clock )
                 clock = clock - clocks

!   if desired, compute the fill in

                 CALL SLS_initialize( solver, data, SLS_control, SLS_inform )
                 CALL SLS_analyse( MAT, data, SLS_control, SLS_inform,         &
                                   PERM = PERM )
                 CALL SLS_terminate( data, SLS_control, SLS_inform )
                 fill = REAL( SLS_inform%entries_in_factors ) / REAL( MAT%ne )
                 WRITE( out, "( A, ' order time = ', A, F0.2,  1X, A,          &
                &     ' analyse status = ', I0, /, ' size(matrix,factor,',     &
                &     '%fill) = ', I0, ', ', I0, ', ', F0.2 )")                &
                   TRIM( pname ), TRIM( STRING_leading_zero( clock ) ), clock, &
                   TRIM( SLS_inform%solver ), SLS_inform%status,               &
                   MAT%ne,SLS_inform%entries_in_factors, fill

!   write a summary to a file

                 IF ( inform%status < 0 ) clock = - clock
                 WRITE( rfiledevice, "( A10, I8, 1X, A3, I4, F10.2, 5I2,       &
                &                       4I4, 3I3, I4, 2I2, 1X, F0.2 )")        &
                   pname, nm, control%version, inform%status, clock,           &
                   control%metis5_ptype, control%metis5_ctype,                 &
                   control%metis5_iptype, control%metis5_rtype,                &
                   control%metis5_compress + 2 * control%metis5_ccorder,       &
                   control%metis5_pfactor, control%metis5_nseps,               &
                   control%metis5_niparts, control%metis5_niter,               &
                   control%metis5_ncuts, control%metis5_minconn,               &
                   control%metis5_contig, control%metis5_ufactor,              &
                   control%metis5_no2hop, control%metis5_dropedges, fill
                END DO
               END DO
              END DO
             END DO
            END DO
           END DO
          END DO
        END IF
        DEALLOCATE( MAT%row, MAT%col, STAT = alloc_stat )
        CLOSE( rfiledevice )

!  ---------------------------------------------------------------
!  conduct an individual test, specifically for the options chosen
!  ---------------------------------------------------------------

      ELSE

        IF ( all_versions ) THEN
          versions = 3
        ELSE
          versions = 1
        END IF

        DO version = 1, versions
          IF ( all_versions ) THEN
            SELECT CASE ( version ) 
            CASE ( 1 )
              control%version = '4.0'
            CASE ( 2 )
              control%version = '5.1'
            CASE ( 3 )
              control%version = '5.2'
            END SELECT
          END IF

!  call the Nodend ordering packages

          CALL CPU_TIME( times ) ; CALL CLOCK_time( clocks )
          CALL NODEND_order_adjacency( nm, K_ptr, K_row, PERM, control, inform )
          CALL CPU_TIME( time ) ; CALL CLOCK_time( clock )
          clock = clock - clocks
          WRITE( out, "( ' Problem: ', A, ' n = ', I0, ', m = ', I0 )" )       &
            TRIM( pname ), n, m
          WRITE( out, "( ' Nodend ', A, ': permutation clock time = ',A,F0.2)")&
            TRIM( control%version ), TRIM( STRING_leading_zero( clock ) ), clock

!   if desired, compute the fill in

          IF ( analyse ) THEN
            CALL SLS_initialize( solver, data, SLS_control, SLS_inform )
            CALL SLS_analyse( MAT, data, SLS_control, SLS_inform, PERM = PERM )
            CALL SLS_terminate( data, SLS_control, SLS_inform )
            fill = REAL( SLS_inform%entries_in_factors ) / REAL( MAT%ne )
            WRITE( out, "( 1X, A, ' analyse status = ', I0, /,                 &
           & ' size(matrix,factor,%fill) = ', I0, ', ', I0, ', ', F0.2 )" )    &
              TRIM( SLS_inform%solver ), SLS_inform%status,                    &
              MAT%ne,SLS_inform%entries_in_factors, fill
          END IF

!   if desired, print the permutation

          IF ( print_permutation )                                             &
            WRITE( out, "( ' permutation = ', /, ( 10I7 ) )" ) PERM

!   if desired, write a summary to a file

          IF ( write_result_summary ) THEN
            BACKSPACE( rfiledevice )
            IF ( inform%status < 0 ) clock = - clock
            IF ( control%version == '4.0' ) THEN
              WRITE( rfiledevice, "( A10, I8, 1X, A3, I4, F10.2, 5I2, 2I4 )",  &
                     advance = 'no' )     &
                pname, nm, control%version, inform%status, clock,              &
                control%metis4_ptype, control%metis4_ctype,                    &
                control%metis4_itype, control%metis4_rtype,                    &
                control%metis4_oflags, control%metis4_pfactor,                 &
                control%metis4_nseps
            ELSE 
              WRITE( rfiledevice,                                              &
                 "( A10, I8, 1X, A3, I4, F10.2, 5I2, 4I4, 3I3, I4, 2I2 )",     &
                    advance = 'no' )                                           &
                pname, nm, control%version, inform%status, clock,              &
                control%metis5_ptype, control%metis5_ctype,                    &
                control%metis5_iptype, control%metis5_rtype,                   &
                control%metis5_compress + 2 * control%metis5_ccorder,          &
                control%metis5_pfactor, control%metis5_nseps,                  &
                control%metis5_niparts, control%metis5_niter,                  &
                control%metis5_ncuts, control%metis5_minconn,                  &
                control%metis5_contig, control%metis5_ufactor,                 &
                control%metis5_no2hop, control%metis5_dropedges
            END IF
            IF ( analyse ) THEN
              WRITE( rfiledevice, "( 1X, F0.2 )" ) fill
            ELSE
              WRITE( rfiledevice, "( '' )" )
            END IF
            IF ( version == versions ) CLOSE( rfiledevice )
          END IF

!   if desired, write the permutation to a file

          IF ( write_permutation ) THEN
            INQUIRE( FILE = pfilename, EXIST = filexx )
            IF ( filexx ) THEN
               OPEN( pfiledevice, FILE = pfilename, FORM = 'FORMATTED',        &
                   STATUS = 'OLD', IOSTAT = iores )
            ELSE
               OPEN( pfiledevice, FILE = pfilename, FORM = 'FORMATTED',        &
                    STATUS = 'NEW', IOSTAT = iores )
            END IF
            IF ( iores /= 0 ) THEN
              write( out, 2010 ) iores, pfilename
              STOP
            END IF

            WRITE( pfiledevice, "( /, ' Problem:    ', A10 )" ) pname
            WRITE( pfiledevice, "( ' permutation = ', /, ( 10I7 ) )" ) PERM
            CLOSE( pfiledevice )
!           DO i = 1, n
!             WRITE( pfiledevice, "( 2I8 )" ) i, PERM( i )
!           END DO
          END IF
        END DO
        DEALLOCATE( MAT%row, MAT%col, STAT = alloc_stat )
      END IF

!  terminate

      IF ( is_specfile ) CLOSE( input_specfile )
      DEALLOCATE(  K_row, K_col, K_ptr, PERM, IW )
      CALL CUTEST_cterminate_r( cutest_status )
      status = 0

      RETURN

 910 CONTINUE
     WRITE( out, "( ' CUTEst error, status = ', i0, ', stopping' )")          &
       cutest_status
     status = - 98
     RETURN

!  Non-executable statements

 2000 FORMAT( ' Allocation error, variable ', A8, ' status = ', I6 )
 2010 FORMAT( ' IOSTAT = ', I6, ' when opening file ', A9, '. Stopping' )

!  End of subroutine USE_NODEND

     END SUBROUTINE USE_NODEND

!  End of module USENODEND

   END MODULE GALAHAD_USENODEND_precision

! control%metis4_ptype                                      0,1    I2
! control%metis4_ctype                                      1-4    I2
! control%metis4_itype                                      1,2    I2
! control%metis4_rtype                                      1,2    I2
! control%metis4_oflags                                     0-3    I2
! control%metis4_pfactor                                    -1-*   I4
! control%metis4_nseps                                      *      I4
! 
! control%metis5_ptype                                      0,1    I2
! control%metis5_ctype                                      0,1    I2
! control%metis5_iptype                                     2,3    I2
! control%metis5_rtype                                      2,3    I2
! control%metis4_oflags = control%metis5_compress + 2*control%metis5_ccorder I2
! control%metis5_pfactor                                    0 *    I4
! control%metis5_nseps                                      *      I4
! 
! control%metis5_niparts                                    *             I4
! control%metis5_niter                                      * 1-999 (say) I4
! control%metis5_ncuts                                      -1            I2
! control%metis5_minconn                                    0,1           I2
! control%metis5_contig                                     0,1           I2
! control%metis5_ufactor                                    1-200         I4
! control%metis5_no2hop                                     0,1           I2
! control%metis5_dropedges                                  0,1           I2




