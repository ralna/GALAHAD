! THIS VERSION: GALAHAD 5.2 - 2025-02-19 AT 10:30 GMT.

#include "galahad_modules.h"
#include "cutest_routines.h"

!-*-*-*-*-*-*-*-  G A L A H A D   U S E M E T I S   M O D U L E  -*-*-*-*-*-*-

!  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
!  Principal authors: Nick Gould and Dominique Orban

!  History -
!   originally released with GALAHAD Version 5.2. February 19th 2025

!  For full documentation, see
!   http://galahad.rl.ac.uk/galahad-www/specs.html

    MODULE GALAHAD_USEMETIS_precision

     USE GALAHAD_KINDS_precision

!    --------------------------------------------
!    | CUTEst/AMPL interface to METIS, a method |
!    | for ordering symmetric sparse matices    |
!    --------------------------------------------

      USE CUTEST_INTERFACE_precision
      USE GALAHAD_CLOCK
      USE GALAHAD_SPECFILE_precision
      USE GALAHAD_SORT_precision, ONLY: SORT_reorder_by_cols
!      USE GALAHAD_METIS_precision
      USE GALAHAD_COPYRIGHT

      IMPLICIT NONE

      PRIVATE
      PUBLIC :: USE_METIS

    CONTAINS

!-*-*-*-*-*-*-*-*-*-   U S E _ M E T I S  S U B R O U T I N E   -*-*-*-*-*-*-*-

     SUBROUTINE USE_METIS( input )

!  --------------------------------------------------------------------
!
!  Solve the linear system from CUTEst
!
!     ( H  A^T ) ( x ) = ( g )
!     ( A   0  ) ( y )   ( c )
!
!  using the symmetric linear solver METIS
!
!  --------------------------------------------------------------------

!  Dummy argument

      INTEGER ( KIND = ip_ ), INTENT( IN ) :: input

!  Parameters

      REAL ( KIND = rp_ ), PARAMETER :: zero = 0.0_rp_
      REAL ( KIND = rp_ ), PARAMETER :: one = 1.0_rp_
      REAL ( KIND = rp_ ), PARAMETER :: two = 2.0_rp_
      REAL ( KIND = rp_ ), PARAMETER :: ten = 10.0_rp_
      REAL ( KIND = rp_ ), PARAMETER :: infinity = ten ** 19
      REAL ( KIND = rp_ ), PARAMETER :: K22 = ten ** 6

!  Scalars

!     INTEGER ( KIND = ip_ ) :: iores, smt_stat
      INTEGER ( KIND = ip_ ) :: status, alloc_stat, cutest_status
      REAL :: time, times
      REAL ( KIND = rp_ ) :: clock, clocks
      LOGICAL :: is_specfile

      INTEGER ( KIND = ip_ ) :: i, k, l, lh, lj, lk, lk2
      INTEGER ( KIND = ip_ ) :: n, m, nm, nm1, nnzh, nnzj, nnzk

      INTEGER ( KIND = ip_ ), PARAMETER :: n_dum = 2
      INTEGER ( KIND = ip_ ), DIMENSION( n_dum + 1 ) :: PTR_dum = (/ 1, 2, 3 /)
      INTEGER ( KIND = ip_ ), DIMENSION( n_dum ) :: ROW_dum = (/ 2, 1 /)
      INTEGER ( KIND = ip_ ), DIMENSION( 1 ) :: PERM_dum, INVP_dum
      INTEGER ( KIND = ip_ ), DIMENSION( 8 ) :: ICNTL_metis

      INTEGER ( KIND = ip_ ), ALLOCATABLE, DIMENSION( : ) :: K_row, K_col, K_ptr
      INTEGER ( KIND = ip_ ), ALLOCATABLE, DIMENSION( : ) :: PERM, INVP, IW

!  Functions

!$    INTEGER ( KIND = ip_ ) :: OMP_GET_MAX_THREADS

!  Specfile characteristics

      INTEGER ( KIND = ip_ ), PARAMETER :: input_specfile = 34
      INTEGER ( KIND = ip_ ), PARAMETER :: lspec = 15
      CHARACTER ( LEN = 16 ) :: specname = 'RUNMETIS'
      TYPE ( SPECFILE_item_type ), DIMENSION( lspec ) :: spec
      CHARACTER ( LEN = 16 ) :: runspec = 'RUNMETIS.SPC'

      INTEGER ( KIND = ip_ ) :: ptype
      INTEGER ( KIND = ip_ ) :: objtype
      INTEGER ( KIND = ip_ ) :: ctype
      INTEGER ( KIND = ip_ ) :: iptype
      INTEGER ( KIND = ip_ ) :: rtype
      INTEGER ( KIND = ip_ ) :: dbglvl
      INTEGER ( KIND = ip_ ) :: niparts
      INTEGER ( KIND = ip_ ) :: niter
      INTEGER ( KIND = ip_ ) :: ncuts
      INTEGER ( KIND = ip_ ) :: seed
      INTEGER ( KIND = ip_ ) :: ondisk
      INTEGER ( KIND = ip_ ) :: minconn
      INTEGER ( KIND = ip_ ) :: contig
      INTEGER ( KIND = ip_ ) :: compress
      INTEGER ( KIND = ip_ ) :: ccorder
      INTEGER ( KIND = ip_ ) :: pfactor
      INTEGER ( KIND = ip_ ) :: nseps
      INTEGER ( KIND = ip_ ) :: ufactor
      INTEGER ( KIND = ip_ ) :: numbering
      INTEGER ( KIND = ip_ ) :: dropedges
      INTEGER ( KIND = ip_ ) :: no2hop
      INTEGER ( KIND = ip_ ) :: twohop
      INTEGER ( KIND = ip_ ) :: fast


!  The default values for METIS could have been set as:

! BEGIN RUNMETIS SPECIFICATIONS (DEFAULT)
!  write-problem-data                                NO
!  problem-data-file-name                            METIS.data
!  problem-data-file-device                          26
!  print-full-solution                               NO
!  write-solution                                    NO
!  solution-file-name                                METISSOL.d
!  solution-file-device                              62
!  write-result-summary                              NO
!  result-summary-file-name                          METISRES.d
!  result-summary-file-device                        47
!  symmetric-linear-equation-solver                  ssids
!  kkt-system                                        YES
!  barrier-perturbation                              1.0
!  solution-passes                                   1
!  solve                                             YES
! END RUNMETIS SPECIFICATIONS

!  Default values for specfile-defined parameters

      INTEGER ( KIND = ip_ ) :: passes = 1
      INTEGER ( KIND = ip_ ) :: dfiledevice = 26
      INTEGER ( KIND = ip_ ) :: sfiledevice = 62
      INTEGER ( KIND = ip_ ) :: rfiledevice = 47
      LOGICAL :: write_problem_data   = .FALSE.
      LOGICAL :: write_solution       = .FALSE.
      LOGICAL :: write_result_summary = .FALSE.
      LOGICAL :: kkt_system = .TRUE.
      LOGICAL :: solve = .TRUE.
      CHARACTER ( LEN = 30 ) :: dfilename = 'METIS.data'
      CHARACTER ( LEN = 30 ) :: sfilename = 'METISSOL.d'
      CHARACTER ( LEN = 30 ) :: rfilename = 'METISRES.d'
      LOGICAL :: fulsol = .FALSE.
      REAL ( KIND = rp_ ) :: barrier_pert = 1.0_rp_

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

!  check to see if the MeTiS ordering packages is available

     CALL galahad_metis_setopt( ICNTL_metis )
     CALL galahad_metis( n_dum, PTR_dum, ROW_dum, 1_ip_, ICNTL_metis,          &
                         INVP_dum, PERM_dum )
     IF ( PERM_dum( 1 ) <= 0 ) THEN
       WRITE( out, "( ' No MeTiS available, stopping' )" ) ; STOP
     END IF

!  ------------------ Open the specfile for runmetis ----------------

      INQUIRE( FILE = runspec, EXIST = is_specfile )
      IF ( is_specfile ) THEN
        OPEN( input_specfile, FILE = runspec, FORM = 'FORMATTED',              &
              STATUS = 'OLD' )

!  define the keywords

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
        spec( 11 )%keyword = ''
        spec( 12 )%keyword = 'kkt-system'
        spec( 13 )%keyword = 'barrier-perturbation'
        spec( 14 )%keyword = 'solution-passes'
        spec( 15 )%keyword = 'solve'

!  read the specfile

        CALL SPECFILE_read( input_specfile, specname, spec, lspec, errout )

!  interpret the result

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
        CALL SPECFILE_assign_logical( spec( 12 ), kkt_system, errout )
        CALL SPECFILE_assign_real( spec( 13 ), barrier_pert, errout )
        CALL SPECFILE_assign_integer( spec( 14 ), passes, errout )
        CALL SPECFILE_assign_logical( spec( 15 ), solve, errout )
      END IF

!  determine the number of variables and constraints

      CALL CUTEST_pname( cutest_status, input, pname )
      IF ( cutest_status /= 0 ) GO TO 910
      CALL CUTEST_cdimen_r( cutest_status, input, n, m )
      IF ( cutest_status /= 0 ) GO TO 910
      WRITE( out, "( ' Problem: ', A, ' n = ', I0, ', m = ', I0 )" )        &
        TRIM( pname ), n, m
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
      IF ( status < 0 ) THEN
        WRITE( out, "( ' sort error = ', I0, ' stopping' )" ) i ; STOP
      END IF

!do i = 1, n
!write(6,"( 'col ', I0, ' row =  ',  /, ( 10I5 ) )" ) i, &
! ( K_row( l ), l = K_ptr( i ), K_ptr( i + 1 ) - 1 )
!end do

!  allocate space for the permutation and its inverse

      ALLOCATE( INVP( nm ), PERM( nm ), STAT = alloc_stat )
      IF ( alloc_stat /= 0 ) THEN
        WRITE( out, 2000 ) 'PERM', alloc_stat ; STOP
      END IF

!  call the MeTiS ordering packages

      CALL CPU_TIME( times ) ; CALL CLOCK_time( clocks )
      CALL galahad_metis_setopt( ICNTL_metis )
      ICNTL_metis( 1 ) = 1
      ICNTL_metis( 5 ) = 2
      CALL galahad_metis( nm, K_ptr, K_row, 1_ip_, ICNTL_metis, INVP, PERM )
      CALL CPU_TIME( time ) ; CALL CLOCK_time( clock )
      WRITE( out, "( ' Metis 5.2 clock time = ', F6.2 )" ) clock - clocks

!  terminate

      IF ( is_specfile ) CLOSE( input_specfile )
      DEALLOCATE(  K_row, K_col, K_ptr, PERM, INVP, IW )
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

!  End of subroutine USE_METIS

     END SUBROUTINE USE_METIS

!  End of module USEMETIS

   END MODULE GALAHAD_USEMETIS_precision
