! THIS VERSION: GALAHAD 5.6 - 2026-08-23 AT 15:10 GMT.

#include "galahad_modules.h"

!-*-*-*-*-*-*-*- G A L A H A D  -  N A G L I B    M O D U L E -*-*-*-*-*-*-*-

  MODULE GALAHAD_NAGLIB_precision

    USE GALAHAD_KINDS_precision, ONLY: ip_, rp_

    IMPLICIT NONE ( TYPE, EXTERNAL )

    PRIVATE
    PUBLIC E04NPF, E04NQF, E04NRF, E04NSF, E04NTF, E04NUF, E04NXF, E04NYF,     &
           X04AAF, X04ACF, X04ADF, E04NQF_transfer_control

!----------------------
!   P a r a m e t e r s
!----------------------

      REAL ( KIND = rp_ ), PARAMETER :: one = 1.0_rp_
      REAL ( KIND = rp_ ), PARAMETER :: ten = 10.0_rp_
      LOGICAL :: print_changes = .FALSE.

!----------------------
!   I n t e r f a c e s
!----------------------

!  interface blocks for Fortran functions

    INTERFACE
      SUBROUTINE E04NQF( start, qphx, m, n, ne, nname, lenc, ncolh, iobj,      &
                         objadd, prob, acol, inda, loca, bl, bu, c, names,     &
                         helast, hs, x, pi, rc, ns, ninf, sinf, obj, cw,       &
                         lencw, iw, leniw, rw, lenrw, cuser, iuser, ruser,     &
                         ifail )
      IMPORT :: ip_, rp_
      INTEGER ( KIND = ip_ ), INTENT( IN ) :: m, n, ne, nname, lenc, ncolh
      INTEGER ( KIND = ip_ ), INTENT( IN ) :: iobj, lencw, leniw, lenrw
      INTEGER ( KIND = ip_ ), INTENT( IN ) :: inda( ne ), loca( n + 1 )
      INTEGER ( KIND = ip_ ), INTENT( IN ) :: helast( n + m )
      INTEGER ( KIND = ip_ ), INTENT( INOUT ) :: hs( n + m ), ns, iw( leniw )
      INTEGER ( KIND = ip_ ), INTENT( INOUT ) :: iuser( * ), ifail
      INTEGER ( KIND = ip_ ), INTENT( OUT ) :: ninf
      REAL ( KIND = rp_ ), INTENT( IN ) :: objadd
      REAL ( KIND = rp_ ), INTENT( INOUT ) :: acol( ne )
      REAL ( KIND = rp_ ), INTENT( INOUT ) :: bl( n + m ), bu( n + m )
      REAL ( KIND = rp_ ), INTENT( INOUT ) :: c( MAX( 1, lenc ) ), x( n + m )
      REAL ( KIND = rp_ ), INTENT( INOUT ) :: rw( lenrw ), ruser( * )
      REAL ( KIND = rp_ ), INTENT( OUT ) :: pi( m ), rc( n + m ), sinf, obj
      CHARACTER ( LEN = 1 ), INTENT ( IN ) :: start
      CHARACTER ( LEN = 8 ), INTENT ( IN ) :: prob, names( nname )
      CHARACTER ( LEN = 8 ), INTENT ( INOUT ) :: cw( lencw ), cuser( * )
      INTERFACE
        SUBROUTINE qphx( ncolh, x, hx, nstate, cuser, iuser, ruser )
        IMPORT :: ip_, rp_
        INTEGER ( KIND = ip_ ), INTENT( IN ) :: ncolh, nstate
        INTEGER ( KIND = ip_ ), INTENT( INOUT ) :: iuser( * )
        REAL ( KIND = rp_ ), INTENT( IN ) :: x( ncolh )
        REAL ( KIND = rp_ ), INTENT( INOUT ) :: ruser( * )
        REAL ( KIND = rp_ ), INTENT( OUT ) :: hx( ncolh )
        CHARACTER ( LEN = 8 ), INTENT( INOUT ) :: cuser( * )
        END SUBROUTINE qphx
      END INTERFACE
      END SUBROUTINE E04NQF

      SUBROUTINE E04NPF( cw, lencw, iw, leniw, rw, lenrw, ifail )
      IMPORT :: ip_, rp_
      INTEGER ( KIND = ip_ ), INTENT( IN ) :: lencw, leniw, lenrw
      INTEGER ( KIND = ip_ ), INTENT( INOUT ) :: ifail
      INTEGER ( KIND = ip_ ), INTENT( OUT ) :: iw( leniw )
      REAL ( KIND = rp_ ), INTENT( OUT ) :: rw( lenrw )
      CHARACTER ( LEN = 8 ), INTENT( OUT ) :: cw( lencw )
      END SUBROUTINE E04NPF

      SUBROUTINE E04NRF( ispecs, cw, iw, rw, ifail )
      IMPORT :: ip_, rp_
      INTEGER ( KIND = ip_ ), INTENT( IN ) :: ispecs
      INTEGER ( KIND = ip_ ), INTENT( INOUT ) :: iw( * ),  ifail
      REAL ( KIND  = rp_ ), INTENT( INOUT ) :: rw( * )
      CHARACTER ( LEN = 8 ), INTENT( INOUT ) :: cw( * )
      END SUBROUTINE E04NRF

      SUBROUTINE E04NSF( string, cw, iw, rw, ifail )
      IMPORT :: ip_, rp_
      CHARACTER ( * ), INTENT( IN ) :: string
      INTEGER ( KIND = ip_ ), INTENT( INOUT ) :: iw( * ),  ifail
      REAL ( KIND  = rp_ ), INTENT( INOUT ) :: rw( * )
      CHARACTER ( LEN = 8 ), INTENT( INOUT ) :: cw( * )
      END SUBROUTINE E04NSF

      SUBROUTINE E04NTF( string, ivalue, cw, iw, rw, ifail )
      IMPORT :: ip_, rp_
      CHARACTER ( * ), INTENT( IN ) :: string
      INTEGER ( KIND = ip_ ), INTENT( IN ) :: ivalue
      INTEGER ( KIND = ip_ ), INTENT( INOUT ) :: iw( * ),  ifail
      REAL ( KIND  = rp_ ), INTENT( INOUT ) :: rw( * )
      CHARACTER ( LEN = 8 ), INTENT( INOUT ) :: cw( * )
      END SUBROUTINE E04NTF

      SUBROUTINE E04NUF( string, rvalue, cw, iw, rw, ifail )
      IMPORT :: ip_, rp_
      CHARACTER ( * ), INTENT( IN ) :: string
      REAL ( KIND  = rp_ ), INTENT( IN ) :: rvalue
      INTEGER ( KIND = ip_ ), INTENT( INOUT ) :: iw( * ),  ifail
      REAL ( KIND  = rp_ ), INTENT( INOUT ) :: rw( * )
      CHARACTER ( LEN = 8 ), INTENT( INOUT ) :: cw( * )
      END SUBROUTINE E04NUF

      SUBROUTINE E04NXF( string, ivalue, cw, iw, rw, ifail )
      IMPORT :: ip_, rp_
      CHARACTER ( * ), INTENT( IN ) :: string
      INTEGER ( KIND = ip_ ), INTENT( OUT ) :: ivalue
      INTEGER ( KIND = ip_ ), INTENT( INOUT ) :: iw( * ),  ifail
      REAL ( KIND  = rp_ ), INTENT( INOUT ) :: rw( * )
      CHARACTER ( LEN = 8 ), INTENT( INOUT ) :: cw( * )
      END SUBROUTINE E04NXF

      SUBROUTINE E04NYF( string, rvalue, cw, iw, rw, ifail )
      IMPORT :: ip_, rp_
      CHARACTER ( * ), INTENT( IN ) :: string
      REAL ( KIND  = rp_ ), INTENT( OUT ) :: rvalue
      INTEGER ( KIND = ip_ ), INTENT( INOUT ) :: iw( * ),  ifail
      REAL ( KIND  = rp_ ), INTENT( INOUT ) :: rw( * )
      CHARACTER ( LEN = 8 ), INTENT( INOUT ) :: cw( * )
      END SUBROUTINE E04NYF

      SUBROUTINE X04AAF( iflag, nerr )
      IMPORT :: ip_
      INTEGER ( KIND = ip_ ), INTENT( IN ) :: iflag
      INTEGER ( KIND = ip_ ), INTENT( INOUT ) :: nerr
      END SUBROUTINE X04AAF

      SUBROUTINE X04ACF( iounit, file, mode, ifail )
      IMPORT :: ip_
      INTEGER ( KIND = ip_ ), INTENT( IN ) :: iounit, mode
      INTEGER ( KIND = ip_ ), INTENT( INOUT ) :: ifail
      CHARACTER (*), INTENT( IN ) :: file
      END SUBROUTINE X04ACF

      SUBROUTINE X04ADF( iounit, ifail )
      IMPORT :: ip_
      INTEGER ( KIND = ip_ ), INTENT( IN ) :: iounit
      INTEGER ( KIND = ip_ ), INTENT( INOUT ) :: ifail
      END SUBROUTINE X04ADF
    END INTERFACE

!-------------------------------------------------
!  D e r i v e d   t y p e   d e f i n i t i o n s
!-------------------------------------------------

!  - - - - - - - - - - - - - - - - - - - - - - -
!   control derived type with component defaults
!  - - - - - - - - - - - - - - - - - - - - - - -

      TYPE, PUBLIC :: E04NQF_control_type

!  number of iterations between checking constraint satisfaction

       INTEGER ( KIND = ip_ ) :: check_frequency = 60

!  strategy used to choose the initial basis matrix during cold starts

       INTEGER ( KIND = ip_ ) :: crash_option = 3

!  threshold value used by the crash procedure to accept basis components

        REAL ( KIND = rp_ ) :: crash_tolerance = 0.1

!  controls whether elastic mode starts automatically when infeasible

       INTEGER ( KIND = ip_ ) :: elastic_mode = 1

!  form of the objective function used in elastic mode

       INTEGER ( KIND = ip_ ) :: elastic_objective = 1

!  scalar weight factor for the infeasibility penalty

        REAL ( KIND = rp_ ) :: elastic_weight = 1.0_rp_

!  specifies the unit for error messages, -ve supresses all error messages

       INTEGER ( KIND = ip_ ) :: error_file = - 1

!  controls anti-cycling procedures during the optimization process

       INTEGER ( KIND = ip_ ) :: expand_frequency = 10000

!  iteration limit before completely refactoring the basis matrix

       INTEGER ( KIND = ip_ ) :: factorization_frequency = 50

!  maximum absolute violation allowed for bound/linear constraints

        REAL ( KIND = rp_ ) :: feasibility_tolerance = ten ** ( - 6 )

!  threshold value above which a bound is treated as infinity

        REAL ( KIND = rp_ ) :: infinite_bound_size = ten ** 19

!  the maximum number of iterations allowed before termination. 
!  -1 leads to the default max( 10000, 10 max( m, n ) )

       INTEGER ( KIND = ip_ ) :: iterations_limit = - 1

!  sparsity threshold controlling pivot behavior during LU factorization

        REAL ( KIND = rp_ ) :: lu_density_tolerance = 0.6_rp_

!  minimum value below which a pivot element is declared singular

        REAL ( KIND = rp_ ) :: lu_singularity_tolerance = 0.0_rp_

!  stability limit for adding elements during initial basis updates

        REAL ( KIND = rp_ ) :: lu_factor_tolerance = 100.0

!  stability limit for modifying factors during product form updates

        REAL ( KIND = rp_ ) :: lu_update_tolerance = 10.0

!  the pivoting strategy used during the LU factorization. Possible
!  values are 1 = partial (default), 2 = complete, 3 = rook

       INTEGER ( KIND = ip_ ) :: lu_pivoting = 1

!  directs the solver to either minimize or maximize the objective function

        LOGICAL :: minimize = .TRUE.

!  directs the solver to simply find a feasible point, ignoring the objective

        LOGICAL :: feasible_point = .FALSE.

!  unit number used to save or load the active basis state

       INTEGER ( KIND = ip_ ) :: new_basis_file = 0

!unit number used to save or the previous active basis state

       INTEGER ( KIND = ip_ ) :: backup_basis_file = 0

!  frequency of generation of the new basis file

       INTEGER ( KIND = ip_ ) :: save_frequency = 100

!  print each optional parameter specification as it is supplied

        LOGICAL :: list = .FALSE.

!  unit numbers used to save or load the active basis state

       INTEGER ( KIND = ip_ ) :: old_basis_file = 0

!  threshold for determining whether the reduced gradient is zero

        REAL ( KIND = rp_ ) :: optimality_tolerance = ten ** ( - 6 )

!  number of segments of entire pricing vector in which pricing operation is 
!  considered, 1 being the entire vector

       INTEGER ( KIND = ip_ ) :: partial_price = 1

!  prevents very small entries from entering the basis selection

        REAL ( KIND = rp_ ) :: pivot_tolerance = ten ** ( - 10 )

!  unit number for the main diagnostic output file, 0 disabled all printing

!      INTEGER ( KIND = ip_ ) :: print_file = 6
       INTEGER ( KIND = ip_ ) :: print_file = 0

!  the depth/verbosity level of internal diagnostic information, 0 = none,
!  1 = yes, 10 = debug

       INTEGER ( KIND = ip_ ) :: print_level = 0

!  the iteration interval at which log entries are written

       INTEGER ( KIND = ip_ ) :: print_frequency = 100

!  the active-set algorithm used to solve the quadratic program in Phase 2.
!  Possible values are 1 = Cholesky (default), 2 = CG, 3 = QN

       INTEGER ( KIND = ip_ ) :: qpsolver = 1 

!  the dimension of the reduce Hessian to be used with the Cholesky QP
!  solver option, - 1 is reset to the default min( 2000, nh + 1, n )

       INTEGER ( KIND = ip_ ) :: reduced_hessian_dimension = - 1

!  the scaling stategy used. Possible values are 0 = no scaling, 1 = scale
!  to make the matrix as close to one as possible, 2 = the same as 1 but
!  taling into account the values of the bounds (the default)

       INTEGER ( KIND = ip_ ) :: scale_option = 2

!  factor in (0,1) that controls how many scaling passes (up to 10) are used 
!  to  equilibrate the matrix

        REAL ( KIND = rp_ ) :: scale_tolerance = 0.9_rp_

!  should the scale factors be written to the output file

        LOGICAL :: scale_print = .FALSE.

!  should the solution be written to the output file

        LOGICAL :: solution = .FALSE.
!       LOGICAL :: solution = .TRUE.

!  the unit number of a separate file to write the solution to

       INTEGER ( KIND = ip_ ) :: solution_file = 0

!  unit number for short, single-line iteration summaries

       INTEGER ( KIND = ip_ ) :: summary_file = 0
!      INTEGER ( KIND = ip_ ) :: summary_file = 6

!  the iteration interval for writing to the summary file

       INTEGER ( KIND = ip_ ) :: summary_frequency = 100

!  a limit on the number of super-basic variables, - 1 is reset to the
!  default min( nh + 1, n )

       INTEGER ( KIND = ip_ ) :: superbasics_limit = - 1

!  prevent printing the entire list of optional settings on initialization

        LOGICAL :: suppress_parameters = .TRUE.

!  prints additional information on the progress of major and minor 
!  iterations, and crash statistics

        LOGICAL :: system_information = .FALSE.

!  If i > 0, some timing information will be output to the main output file

       INTEGER ( KIND = ip_ ) :: timing_level = 0

!  the maximum step size that is allowed before the step is considered
!  to be infinte

        REAL ( KIND = rp_ ) :: unbounded_step_size = ten ** 19

      END TYPE E04NQF_control_type

!  - - - - - - - - - - - - - - - - - - - - - - -
!   inform derived type with component defaults
!  - - - - - - - - - - - - - - - - - - - - - - -

      TYPE, PUBLIC :: E04NQF_inform_type

!  return status

        INTEGER ( KIND = ip_ ) :: status = 0

!  return status from E04NQF

        INTEGER ( KIND = ip_ ) :: ifail = 0

!  the status of the last attempted allocation/deallocation

        INTEGER ( KIND = ip_ ) :: alloc_status = 0

!  the name of the array for which an allocation/deallocation error ocurred

        CHARACTER ( LEN = 80 ) :: bad_alloc = REPEAT( ' ', 80 )

!  the total number of iterations required

        INTEGER ( KIND = ip_ ) :: iter = - 1

!  the total number of major iterations required

        INTEGER ( KIND = ip_ ) :: major_iter = - 1

!  the total number of factorizations performed

        INTEGER ( KIND = ip_ ) :: nfacts = - 1

!  the final number of superbasics

        INTEGER ( KIND = ip_ ) :: ns = - 1

!  the number of infeasibilities

        INTEGER ( KIND = ip_ ) :: ninf = - 1

!  the value of the objective function at the best estimate of the solution
!   determined by E04NQF

        REAL ( KIND = rp_ ) :: obj = HUGE( one )

!  the sum of the scaled infeasibilities

        REAL ( KIND = rp_ ) :: sinf = HUGE( one )

      END TYPE E04NQF_inform_type

    CONTAINS

!-*-*-   E 0 4 N Q F _ T R A N S F E R _ C O N T R O L  S U B R O U T I N E  -*-

      SUBROUTINE E04NQF_transfer_control( control, cw, lencw, iw, leniw,       &
                                          rw, lenrw, ifail )
      INTEGER ( KIND = ip_ ), INTENT( IN ) :: lencw, leniw, lenrw
      INTEGER ( KIND = ip_ ), INTENT( INOUT ) :: iw( leniw ),  ifail
      REAL ( KIND  = rp_ ), INTENT( INOUT ) :: rw( lenrw )
      CHARACTER ( LEN = 8 ), INTENT( INOUT ) :: cw( lencw )
      TYPE ( E04NQF_control_type ) :: control

!  transfer control components to their E04NQF equivalents

      CALL E04NTF_transfer_control( 'Check Frequency',                         &
                                    control%check_frequency,                   &
                                    cw, lencw, iw, leniw, rw, lenrw, ifail )
      CALL E04NTF_transfer_control( 'Crash Option',                            &
                                    control%crash_option,                      &
                                    cw, lencw, iw, leniw, rw, lenrw, ifail )
      CALL E04NUF_transfer_control( 'Crash Tolerance',                         &
                                    control%crash_tolerance,                   &
                                    cw, lencw, iw, leniw, rw, lenrw, ifail )
      CALL E04NTF_transfer_control( 'Elastic Mode',                            &
                                    control%elastic_mode,                      &
                                    cw, lencw, iw, leniw, rw, lenrw, ifail )
      CALL E04NTF_transfer_control( 'Elastic Objective',                       &
                                    control%elastic_objective,                 &
                                    cw, lencw, iw, leniw, rw, lenrw, ifail )
      CALL E04NUF_transfer_control( 'Elastic Weight',                          &
                                    control%elastic_weight,                    &
                                    cw, lencw, iw, leniw, rw, lenrw, ifail )
      CALL E04NTF_transfer_control( 'Expand Frequency',                        &
                                    control%expand_frequency,                  &
                                    cw, lencw, iw, leniw, rw, lenrw, ifail )
      CALL E04NTF_transfer_control( 'Factorization Frequency',                 &
                                    control%factorization_frequency,           &
                                    cw, lencw, iw, leniw, rw, lenrw, ifail )
      CALL E04NUF_transfer_control( 'Feasibility Tolerance',                   &
                                    control%feasibility_tolerance,             &
                                    cw, lencw, iw, leniw, rw, lenrw, ifail )
      CALL E04NUF_transfer_control( 'Infinite Bound Size',                     &
                                    control%infinite_bound_size,               &
                                    cw, lencw, iw, leniw, rw, lenrw, ifail )
      CALL E04NTF_transfer_control( 'Iterations Limit',                        &
                                    control%iterations_limit,                  &
                                    cw, lencw, iw, leniw, rw, lenrw, ifail )
      CALL E04NUF_transfer_control( 'LU Density Tolerance',                    &
                                    control%lu_density_tolerance,              &
                                    cw, lencw, iw, leniw, rw, lenrw, ifail )
      CALL E04NUF_transfer_control( 'LU Singularity Tolerance',                &
                                    control%lu_singularity_tolerance,          &
                                    cw, lencw, iw, leniw, rw, lenrw, ifail )
      CALL E04NUF_transfer_control( 'LU Factor Tolerance',                     &
                                    control%lu_factor_tolerance,               &
                                    cw, lencw, iw, leniw, rw, lenrw, ifail )
      CALL E04NUF_transfer_control( 'LU Update Tolerance',                     &
                                    control%lu_update_tolerance,               &
                                    cw, lencw, iw, leniw, rw, lenrw, ifail )
      IF (  control%lu_pivoting == 2 ) THEN
        CALL E04NSF_transfer_control( 'LU Complete Pivoting',                  &
                                      cw, lencw, iw, leniw, rw, lenrw, ifail )
      ELSE IF (  control%lu_pivoting == 3 ) THEN
        CALL E04NSF_transfer_control( 'LU Rook Pivoting',                      &
                                      cw, lencw, iw, leniw, rw, lenrw, ifail )
      ELSE
        CALL E04NSF_transfer_control( 'LU Partial Pivoting',                   &
                                      cw, lencw, iw, leniw, rw, lenrw, ifail )
      END IF
      IF ( control%feasible_point ) THEN
        CALL E04NSF_transfer_control( 'Feasible Point',                        &
                                      cw, lencw, iw, leniw, rw, lenrw, ifail )
      ELSE IF ( control%minimize ) THEN
        CALL E04NSF_transfer_control( 'Minimize',                              &
                                      cw, lencw, iw, leniw, rw, lenrw, ifail )
      ELSE
        CALL E04NSF_transfer_control( 'Maximize',                              &
                                     cw, lencw, iw, leniw, rw, lenrw, ifail )
      END IF
      CALL E04NTF_transfer_control( 'New Basis File',                          &
                                    control%new_basis_file,                    &
                                    cw, lencw, iw, leniw, rw, lenrw, ifail )
      CALL E04NTF_transfer_control( 'Backup Basis File',                       &
                                    control%backup_basis_file,                 &
                                    cw, lencw, iw, leniw, rw, lenrw, ifail )
      CALL E04NTF_transfer_control( 'Save Frequency',                          &
                                    control%save_frequency,                    &
                                    cw, lencw, iw, leniw, rw, lenrw, ifail )
      IF ( control%list ) THEN
        CALL E04NSF_transfer_control( 'List',                                  &
                                      cw, lencw, iw, leniw, rw, lenrw, ifail )
      ELSE
        CALL E04NSF_transfer_control( 'Nolist',                                &
                                      cw, lencw, iw, leniw, rw, lenrw, ifail )
      END IF
      CALL E04NTF_transfer_control( 'Old Basis File',                          &
                                    control%old_basis_file,                    &
                                    cw, lencw, iw, leniw, rw, lenrw, ifail )
      CALL E04NUF_transfer_control( 'Optimality Tolerance',                    &
                                    control%optimality_tolerance,              &
                                    cw, lencw, iw, leniw, rw, lenrw, ifail )
      CALL E04NTF_transfer_control( 'Partial Price',                           &
                                    control%partial_price,                     &
                                    cw, lencw, iw, leniw, rw, lenrw, ifail )
      CALL E04NUF_transfer_control( 'Pivot Tolerance',                         &
                                    control%pivot_tolerance,                   &
                                    cw, lencw, iw, leniw, rw, lenrw, ifail )
      CALL E04NTF_transfer_control( 'Print File',                              &
                                    control%print_file,                        &
                                    cw, lencw, iw, leniw, rw, lenrw, ifail )
      CALL E04NTF_transfer_control( 'Print Level',                             &
                                    control%print_level,                       &
                                    cw, lencw, iw, leniw, rw, lenrw, ifail )
      CALL E04NTF_transfer_control( 'Print Frequency',                         &
                                    control%print_frequency,                   &
                                    cw, lencw, iw, leniw, rw, lenrw, ifail )
      IF ( control%qpsolver == 2 ) THEN
        CALL E04NSF_transfer_control( 'QPSolver CG',                           &
                                      cw, lencw, iw, leniw, rw, lenrw, ifail )
      ELSE IF ( control%qpsolver == 3 ) THEN
        CALL E04NSF_transfer_control( 'QPSolver QN',                           &
                                      cw, lencw, iw, leniw, rw, lenrw, ifail )
      ELSE
        CALL E04NSF_transfer_control( 'QPSolver Cholesky',                     &
                                      cw, lencw, iw, leniw, rw, lenrw, ifail )
      END IF
      CALL E04NTF_transfer_control( 'Reduced Hessian Dimension',               &
                                    control%reduced_hessian_dimension,         &
                                    cw, lencw, iw, leniw, rw, lenrw, ifail )
      CALL E04NTF_transfer_control( 'Scale Option',                            &
                                    control%scale_option,                      &
                                    cw, lencw, iw, leniw, rw, lenrw, ifail )
      CALL E04NUF_transfer_control( 'Scale Tolerance',                         &
                                    control%scale_tolerance,                   &
                                    cw, lencw, iw, leniw, rw, lenrw, ifail )
      IF ( control%scale_print )                                               &
        CALL E04NSF_transfer_control( 'Scale Print',                           &
                                      cw, lencw, iw, leniw, rw, lenrw, ifail )
      IF ( control%solution ) THEN
        CALL E04NSF_transfer_control( 'Solution Yes',                          &
                                      cw, lencw, iw, leniw, rw, lenrw, ifail )
      ELSE
        CALL E04NSF_transfer_control( 'Solution No',                           &
                                      cw, lencw, iw, leniw, rw, lenrw, ifail )
      END IF
      CALL E04NTF_transfer_control( 'Solution File',                           &
                                    control%solution_file,                     &
                                    cw, lencw, iw, leniw, rw, lenrw, ifail )
      CALL E04NTF_transfer_control( 'Summary File',                            &
                                    control%summary_file,                      &
                                    cw, lencw, iw, leniw, rw, lenrw, ifail )
      CALL E04NTF_transfer_control( 'Summary Frequency',                       &
                                    control%summary_frequency,                 &
                                    cw, lencw, iw, leniw, rw, lenrw, ifail )
      CALL E04NTF_transfer_control( 'Superbasics Limit',                       &
                                    control%superbasics_limit,                 &
                                    cw, lencw, iw, leniw, rw, lenrw, ifail )
      IF ( control%suppress_parameters )                                       &
        CALL E04NSF_transfer_control( 'Suppress Parameters',                   &
                                      cw, lencw, iw, leniw, rw, lenrw, ifail )
      IF ( control%system_information ) THEN
        CALL E04NSF_transfer_control( 'System Information Yes',                &
                                      cw, lencw, iw, leniw, rw, lenrw, ifail )
      ELSE
        CALL E04NSF_transfer_control( 'System Information No',                 &
                                      cw, lencw, iw, leniw, rw, lenrw, ifail )
      END IF
      CALL E04NTF_transfer_control( 'Timing Level',                            &
                                    control%timing_level,                      &
                                    cw, lencw, iw, leniw, rw, lenrw, ifail )
      CALL E04NUF_transfer_control( 'Unbounded Step Size',                     &
                                    control%unbounded_step_size,               &
                                    cw, lencw, iw, leniw, rw, lenrw, ifail )

      RETURN

!  end of subroutine E04NQF_transfer_control

      END SUBROUTINE E04NQF_transfer_control

!-*-*-   E 0 4 N S F _ T R A N S F E R _ C O N T R O L  S U B R O U T I N E  -*-

      SUBROUTINE E04NSF_transfer_control( string, cw, lencw,                   &
                                          iw, leniw, rw, lenrw, ifail )
      CHARACTER ( * ), INTENT( IN ) :: string
      INTEGER ( KIND = ip_ ), INTENT( IN ) :: lencw, leniw, lenrw
      INTEGER ( KIND = ip_ ), INTENT( INOUT ) :: iw( leniw ),  ifail
      REAL ( KIND  = rp_ ), INTENT( INOUT ) :: rw( lenrw )
      CHARACTER ( LEN = 8 ), INTENT( INOUT ) :: cw( lencw )

!  local variables

      INTEGER ( KIND = ip_ ) :: iw_old( leniw )
      REAL ( KIND  = rp_ ) :: rw_old( lenrw )
      CHARACTER ( LEN = 8 ) :: cw_old( lencw )

      IF ( print_changes ) THEN
        iw_old = iw ; rw_old = rw ; cw_old = cw
      END IF
      CALL E04NSF( string, cw, iw, rw, ifail )
      IF ( print_changes ) THEN
        WRITE( 6, "( A, ' added' )" ) string
        CALL E04NQF_transfer_changes( cw, cw_old, lencw, iw, iw_old, leniw,    &
                                      rw, rw_old, lenrw )
      END IF
      RETURN

!  end of subroutine E04NSF_transfer_control

      END SUBROUTINE E04NSF_transfer_control

!-*-*-   E 0 4 N T F _ T R A N S F E R _ C O N T R O L  S U B R O U T I N E  -*-

      SUBROUTINE E04NTF_transfer_control( string, ivalue, cw, lencw,           &
                                          iw, leniw, rw, lenrw, ifail )
      CHARACTER ( * ), INTENT( IN ) :: string
      INTEGER ( KIND = ip_ ), INTENT( IN ) :: ivalue
      INTEGER ( KIND = ip_ ), INTENT( IN ) :: lencw, leniw, lenrw
      INTEGER ( KIND = ip_ ), INTENT( INOUT ) :: iw( leniw ),  ifail
      REAL ( KIND  = rp_ ), INTENT( INOUT ) :: rw( lenrw )
      CHARACTER ( LEN = 8 ), INTENT( INOUT ) :: cw( lencw )

!  local variables

      INTEGER ( KIND = ip_ ) :: iw_old( leniw )
      REAL ( KIND  = rp_ ) :: rw_old( lenrw )
      CHARACTER ( LEN = 8 ) :: cw_old( lencw )

      IF ( print_changes ) THEN
        iw_old = iw ; rw_old = rw ; cw_old = cw
      END IF
      CALL E04NTF( string, ivalue, cw, iw, rw, ifail )
      IF ( print_changes ) THEN
        WRITE( 6, "( A, ' added with value ', I0 )" ) string, ivalue
        CALL E04NQF_transfer_changes( cw, cw_old, lencw, iw, iw_old, leniw,    &
                                      rw, rw_old, lenrw )
      END IF
      RETURN

!  end of subroutine E04NTF_transfer_control

      END SUBROUTINE E04NTF_transfer_control

!-*-*-   E 0 4 N U F _ T R A N S F E R _ C O N T R O L  S U B R O U T I N E  -*-

      SUBROUTINE E04NUF_transfer_control( string, rvalue, cw, lencw,           &
                                          iw, leniw, rw, lenrw, ifail )
      CHARACTER ( * ), INTENT( IN ) :: string
      REAL ( KIND  = rp_ ), INTENT( IN ) :: rvalue
      INTEGER ( KIND = ip_ ), INTENT( IN ) :: lencw, leniw, lenrw
      INTEGER ( KIND = ip_ ), INTENT( INOUT ) :: iw( leniw ),  ifail
      REAL ( KIND  = rp_ ), INTENT( INOUT ) :: rw( lenrw )
      CHARACTER ( LEN = 8 ), INTENT( INOUT ) :: cw( lencw )

!  local variables

      INTEGER ( KIND = ip_ ) :: iw_old( leniw )
      REAL ( KIND  = rp_ ) :: rw_old( lenrw )
      CHARACTER ( LEN = 8 ) :: cw_old( lencw )

      IF ( print_changes ) THEN
        iw_old = iw ; rw_old = rw ; cw_old = cw
      END IF
      CALL E04NUF( string, rvalue, cw, iw, rw, ifail )
      IF ( print_changes ) THEN
        WRITE( 6, "( A, ' added with value ', G0 )" ) string, rvalue
        CALL E04NQF_transfer_changes( cw, cw_old, lencw, iw, iw_old, leniw,    &
                                      rw, rw_old, lenrw )
      END IF
      RETURN

!  end of subroutine E04NUF_transfer_control

      END SUBROUTINE E04NUF_transfer_control

!-*-*-   E 0 4 N Q F _ T R A N S F E R _ C H A N G E S  S U B R O U T I N E  -*-

      SUBROUTINE E04NQF_transfer_changes( cw, cw_old, lencw, iw, iw_old,       &
                                          leniw, rw, rw_old, lenrw )
      INTEGER ( KIND = ip_ ), INTENT( IN ) :: lencw, leniw, lenrw
      INTEGER ( KIND = ip_ ), INTENT( IN ) :: iw( leniw ), iw_old( leniw )
      REAL ( KIND  = rp_ ), INTENT( IN ) :: rw( lenrw ), rw_old( lenrw )
      CHARACTER ( LEN = 8 ), INTENT( IN ) :: cw( lencw ), cw_old( lencw )

!  local variables

      INTEGER ( KIND = ip_ ) :: i
      DO i = 1, leniw
        IF ( iw( i ) /= iw_old( i ) )                                          &
          WRITE( 6, "( ' iw(', I0, ') changed from ', I0, ' to ', I0 )" )      &
            i, iw_old( i ), iw( i ) 
      END DO
      DO i = 1, lenrw
        IF ( rw( i ) /= rw_old( i ) )                                          &
          WRITE( 6, "( ' rw(', I0, ') changed from ', G0, ' to ', G0 )" )      &
            i, rw_old( i ), rw( i ) 
      END DO
      DO i = 1, lencw
        IF ( cw( i ) /= cw_old( i ) )                                          &
          WRITE( 6, "( ' cw(', I0, ') changed from ', A, ' to ', A )" )        &
            i, TRIM( cw_old( i ) ), TRIM( cw( i ) )
      END DO
      RETURN

!  end of subroutine E04NQF_transfer_changes

      END SUBROUTINE E04NQF_transfer_changes

!  E04NQF.SPC

!   Check Frequency             = 60
!   Crash Option                = 3
!   Crash Tolerance             = 0.1
!   Dump File                   = 0
!   Load File                   = 0
!   Elastic Mode                = 1
!   Elastic Objective           = 1
!   Elastic Weight              = 1.0
!   Expand Frequency            = 10000
!   Factorization Frequency     = 50
!   Feasibility Tolerance       = 1.0E-6
!   Infinite Bound Size         = 1.0E+19
!   Iterations Limit            = -1
!   LU Density Tolerance        = 0.6
!   LU Singularity Tolerance    = 0.0
!   LU Factor Tolerance         = 100.0
!   LU Update Tolerance         = 10.0
!   LU Partial Pivoting
! * LU Complete Pivoting
! * LU Rook Pivoting
!   Minimize
! * Maximize
! * Feasible Point
!   New Basis File              = 0
!   Backup Basis File           = 0
!   Save Frequency              = 100
! * Nolist
! * List
!   Old Basis File              = 0
!   Optimality Tolerance        = 1.0E-6
!   Partial Price               = 1
!   Pivot Tolerance             = 1.0E-10
!   Print File                  = 0
!   Print Level                 = 1
!   Print Frequency             = 100
!   QPSolver Cholesky
! * QPSolver CG
! * QPSolver QN
!   Reduced Hessian Dimension   = ​2000
!   Scale Option                = 2
!   Scale Tolerance             = 0.9
!   Scale Print
!   Solution Yes
! * Solution No
!   Solution File               = 0
!   Summary File                = 0
!   Summary Frequency           = 100
! * Superbasics Limit           = ​MIN( nh + 1, n )
!   Suppress Parameters Off                       
!   System Information No
! * System Information Yes
!   Timing Level                = 0
!   Unbounded Step Size         = 0.0

!  values of iw, rw used prior to solution:
!  ----------------------------------------

!  Print File added with value 6
!   iw(12) changed from 0 to 6
!  Summary File added with value 6
!   iw(13) changed from 0 to 6
!  Reduced Hessian Dimension added with value -1
!   iw(52) changed from -11111 to -1
!  Superbasics Limit added with value -1
!   iw(53) changed from -11111 to -1
!  QPSolver Cholesky added
!   iw(55) changed from -11111 to 0    1 for CG    2 for QN
!  Elastic Mode added with value 1
!   iw(56) changed from -11111 to 1
!  Check Frequency added with value 60
!   iw(58) changed from -11111 to 60
!  Factorization Frequency added with value 50
!   iw(59) changed from -11111 to 50
!  Save Frequency added with value 100
!   iw(60) changed from -11111 to 100
!  Print Frequency added with value 100
!   iw(61) changed from -11111 to 100
!  Summary Frequency added with value 100
!   iw(62) changed from -11111 to 100
!  Expand Frequency added with value 10000
!   iw(63) changed from -11111 to 10000
!  System Information No added
!   iw(71) changed from -11111 to 0     1 for yes
!  Elastic Objective added with value 1
!   iw(73) changed from -11111 to 1
!  Scale Option added with value 2
!   iw(75) changed from -11111 to 2
!  LU Partial Pivoting added
!   iw(80) changed from -11111 to 0     1 for rook  2 for complete
!  Suppress Parameters added
!   iw(81) changed from -11111 to 0
!  Solution Yes added
!   iw(84) changed from -11111 to 2     0 for no
!  Minimize added
!   iw(87) changed from -11111 to 1     0 for feasible point   -1 for maximize
!  Crash Option added with value 3
!   iw(88) changed from -11111 to 3
!  Iterations Limit added with value -1
!   iw(89) changed from -11111 to -1
!  Print Level added with value 1
!   iw(93) changed from -11111 to 1
!  Partial Price added with value 1
!   iw(94) changed from -11111 to 1
!  Backup Basis File added with value 0
!   iw(120) changed from -11111 to 0
!  New Basis File added with value 0
!   iw(124) changed from -11111 to 0
!  Old Basis File added with value 0
!   iw(126) changed from -11111 to 0
!  Solution File added with value 0
!   iw(131) changed from -11111 to 0
!  Timing Level added with value 1
!   iw(182) changed from 0 to 1
!  List added
!   iw(502) changed from 0 to 1        0 for no list
!
!  Pivot Tolerance added with value 0.10000000000000000E-9
!   rw(60) changed from -11111.000000000000 to 0.10000000000000000E-9
!  Optimality Tolerance added with value 0.99999999999999995E-6
!   rw(53) changed from -11111.000000000000 to 0.99999999999999995E-6
!  Feasibility Tolerance added with value 0.99999999999999995E-6
!   rw(56) changed from -11111.000000000000 to 0.99999999999999995E-6
!  Crash Tolerance added with value 0.10000000149011612
!   rw(62) changed from -11111.000000000000 to 0.10000000100000001
!  LU Factor Tolerance added with value 100.00000000000000
!   rw(66) changed from -11111.000000000000 to 100.00000000000000
!  LU Update Tolerance added with value 10.000000000000000
!   rw(67) changed from -11111.000000000000 to 10.000000000000000
!  Infinite Bound Size added with value 0.10000000000000000E+20
!   rw(70) changed from -11111.000000000000 to 0.10000000000000000E+20
!  Unbounded Step Size added with value 0.10000000000000000E+20
!   rw(72) changed from -11111.000000000000 to 0.10000000000000000E+20
!  Elastic Weight added with value 1.0000000000000000
!   rw(88) changed from -11111.000000000000 to 1.0000000000000000
!  Scale Tolerance added with value 0.90000000000000002
!   rw(92) changed from -11111.000000000000 to 0.90000000000000002
!  LU Singularity Tolerance added with value 0.0000000000000000
!   rw(154) changed from -11111.000000000000 to 0.0000000000000000
!   rw(155) changed from -11111.000000000000 to 0.0000000000000000
!  LU Density Tolerance added with value 0.59999999999999998
!   rw(158) changed from -11111.000000000000 to 0.59999999999999998

END MODULE GALAHAD_NAGLIB_precision
