! THIS VERSION: GALAHAD 5.6 - 2026-10-12 AT 12:00 GMT.

#include "galahad_modules.h"

!-*-*-*-*-*-*-*- G A L A H A D  -  N A G L I B    M O D U L E -*-*-*-*-*-*-*-

  MODULE GALAHAD_NAGLIB_precision

    USE GALAHAD_KINDS_precision, ONLY: ip_, rp_, i4_
    USE GALAHAD_SYMBOLS, ONLY: ACTIVE => GALAHAD_ACTIVE,                       &
                               GALAHAD_ok,                                     &
                               GALAHAD_error_allocate,                         &
                               GALAHAD_error_deallocate,                       &
                               GALAHAD_error_restrictions,                     &
                               GALAHAD_error_primal_infeasible,                &
                               GALAHAD_error_factorization,                    &
                               GALAHAD_error_unbounded,                        &
                               GALAHAD_error_ill_conditioned,                  &
                               GALAHAD_no_progress,                            &
                               GALAHAD_error_tiny_step,                        &
                               GALAHAD_error_max_iterations,                   &
                               GALAHAD_error_inertia,                          &
                               GALAHAD_error_file,                             &
                               GALAHAD_error_io,                               &
                               GALAHAD_error_naglib,                           &
                               GALAHAD_unavailable_solver,                     &
                               GALAHAD_no_licence
    USE GALAHAD_SPACE_precision, ONLY: SPACE_resize_array, SPACE_dealloc_array
    USE GALAHAD_SPECFILE_precision, ONLY: SPECFILE_item_type, SPECFILE_read,   &
                                          SPECFILE_assign_value
    USE GALAHAD_SMT_precision, ONLY: SMT_type, SMT_get, SMT_put
    USE GALAHAD_QPT_precision, ONLY: QPT_problem_type, QPT_keyword_A,          &
                                     QPT_keyword_H, QPT_summarize_problem
    USE GALAHAD_MOP_precision, ONLY: MOP_Ax
    USE GALAHAD_CONVERT_precision, ONLY: CONVERT_control_type,                 &
                                         CONVERT_inform_type,                  &
                                         CONVERT_to_sparse_column_format,      &
                                         CONVERT_to_sparse_symmetric_row_format

    IMPLICIT NONE ( TYPE, EXTERNAL )

    PRIVATE
    PUBLIC E04NPF, E04NQF, E04NRF, E04NSF, E04NTF, E04NUF, E04NXF, E04NYF,     &
           X04AAF, X04ACF, X04ADF,  E04NQF_initialize, E04NQF_read_specfile,   &
           E04NQF_solve, E04NQF_terminate, E04NQF_transfer_control,            &
           QPT_problem_type, SMT_type, SMT_get, SMT_put

!----------------------
!   I n t e r f a c e s
!----------------------

    INTERFACE E04NQF_initialize
      MODULE PROCEDURE E04NQF_initialize, E04NQF_full_initialize
    END INTERFACE E04NQF_initialize

    INTERFACE E04NQF_terminate
      MODULE PROCEDURE E04NQF_terminate, E04NQF_full_terminate
    END INTERFACE E04NQF_terminate

!----------------------
!   P a r a m e t e r s
!----------------------

    REAL ( KIND = rp_ ), PARAMETER :: zero = 0.0_rp_
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

!   error and warning diagnostics occur on stream error (-ve = no diagnostics)

        INTEGER ( KIND = ip_ ) :: error = 6

!   general output occurs on stream out

        INTEGER ( KIND = ip_ ) :: out = 6

!  the depth/verbosity level of internal diagnostic information, 0 = none,
!  1 = yes, 10 = debug

        INTEGER ( KIND = ip_ ) :: print_level = 0

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

!       INTEGER ( KIND = ip_ ) :: print_file = 6
        INTEGER ( KIND = ip_ ) :: print_file = 0

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
!       INTEGER ( KIND = ip_ ) :: summary_file = 6

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

!   if %space_critical true, every effort will be made to use as little
!     space as possible. This may result in longer computation time

        LOGICAL :: space_critical = .FALSE.

!   if %deallocate_error_fatal is true, any array/pointer deallocation error
!     will terminate execution. Otherwise, computation will continue

        LOGICAL :: deallocate_error_fatal = .FALSE.

!  all output lines will be prefixed by %prefix(2:LEN(TRIM(%prefix))-1)
!   where %prefix contains the required string enclosed in
!   quotes, e.g. "string" or 'string'

        CHARACTER ( LEN = 30 ) :: prefix = '""                            '

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

!  the value of the primal infeasibility

        REAL ( KIND = rp_ ) :: primal_infeasibility = HUGE( one )

!  the value of the dual infeasibility

        REAL ( KIND = rp_ ) :: dual_infeasibility = HUGE( one )

      END TYPE E04NQF_inform_type

!  - - - - - - - - - - - - - - -
!   data derived type for E04NQF
!  - - - - - - - - - - - - - - -

      TYPE, PUBLIC :: E04NQF_data_type
        INTEGER ( KIND = ip_ ), ALLOCATABLE, DIMENSION( : ) :: HELAST, HS
        INTEGER ( KIND = ip_ ), ALLOCATABLE, DIMENSION( : ) :: I_w, I_user
        REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : ) :: B_l, B_u
        REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : ) :: X, Z, R_w, R_user
        CHARACTER ( LEN = 8 ), ALLOCATABLE, DIMENSION( : )  :: C_w, C_user
        LOGICAL :: original_a, original_h
        LOGICAL :: new_structure = .TRUE.
        TYPE ( SMT_type ) :: A, H
      END TYPE E04NQF_data_type

!  - - - - - - - - - - - -
!   full_data derived type
!  - - - - - - - - - - - -

      TYPE, PUBLIC :: E04NQF_full_data_type
        LOGICAL :: f_indexing = .TRUE.
        TYPE ( E04NQF_data_type ) :: E04NQF_data
        TYPE ( E04NQF_control_type ) :: E04NQF_control
        TYPE ( E04NQF_inform_type ) :: E04NQF_inform
        TYPE ( QPT_problem_type ) :: prob
      END TYPE E04NQF_full_data_type

    CONTAINS
!-*-*-*-*-*-   B Q P D _ I N I T I A L I Z E   S U B R O U T I N E   -*-*-*-*-*

      SUBROUTINE E04NQF_initialize( data, control, inform )

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
!
!  Default control data for E04NQF. This routine should be called before
!  E04NQF_solve
!
!  ---------------------------------------------------------------------------
!
!  Arguments:
!
!  data     private internal data
!  control  a structure containing control information. See preamble
!  inform   a structure containing output information. See preamble
!
! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

      TYPE ( E04NQF_data_type ), INTENT( INOUT ) :: data
      TYPE ( E04NQF_control_type ), INTENT( OUT ) :: control
      TYPE ( E04NQF_inform_type ), INTENT( OUT ) :: inform

      inform%status = GALAHAD_ok

      RETURN

!  End of E04NQF_initialize

      END SUBROUTINE E04NQF_initialize

!- G A L A H A D -  B Q P D _ F U L L _ I N I T I A L I Z E  S U B R O U T I N E

     SUBROUTINE E04NQF_full_initialize( data, control, inform )

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!   Provide default values for E04NQF controls

!   Arguments:

!   data     private internal data
!   control  a structure containing control information. See preamble
!   inform   a structure containing output information. See preamble

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     TYPE ( E04NQF_full_data_type ), INTENT( INOUT ) :: data
     TYPE ( E04NQF_control_type ), INTENT( OUT ) :: control
     TYPE ( E04NQF_inform_type ), INTENT( OUT ) :: inform

     CALL E04NQF_initialize( data%E04NQF_data, control, inform )

     RETURN

!  End of subroutine E04NQF_full_initialize

     END SUBROUTINE E04NQF_full_initialize

!-*-*-*-*-   B Q P D _ R E A D _ S P E C F I L E  S U B R O U T I N E   -*-*-*-

      SUBROUTINE E04NQF_read_specfile( control, device, alt_specname )

!  Reads the content of a specification file, and performs the assignment of
!  values associated with given keywords to the corresponding control parameters

!  The defauly values as given by E04NQF_initialize could (roughly)
!  have been set as:

! BEGIN E04NQF SPECIFICATIONS (DEFAULT)
!  error-printout-device                             6
!  printout-device                                   6
!  print-level                                       0
!  max-levels                                        100
!  spike-space                                       100000
!  max-refinements                                   2
!  min-iterations-before-refinement                  3
!  max-restarts                                      2
!  max-refactorization-interval                      500
!  minimum-objective-before-unbounded                -1.0D+32
!  accuracy-requied                                  1.0D-12
!  max-allowable-error                               1.0D-4
!  space-critical                                    F
!  deallocate-error-fatal                            F
!  output-line-prefix                                ""
! END E04NQF SPECIFICATIONS (DEFAULT)

!  Dummy arguments

      TYPE ( E04NQF_control_type ), INTENT( INOUT ) :: control
      INTEGER ( KIND = ip_ ), INTENT( IN ) :: device
      CHARACTER( LEN = * ), OPTIONAL :: alt_specname

!  Programming: Nick Gould and Ph. Toint, January 2002.

!  Local variables

      INTEGER ( KIND = ip_ ), PARAMETER :: error = 1
      INTEGER ( KIND = ip_ ), PARAMETER :: out = error + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: print_level = out + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: check_frequency = print_level + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: crash_option = check_frequency + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: crash_tolerance = crash_option + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: elastic_mode = crash_tolerance + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: elastic_objective =elastic_mode + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: elastic_weight                      &
                                             = elastic_objective + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: expand_frequency = elastic_weight + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: factorization_frequency             &
                                             = expand_frequency + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: feasibility_tolerance               &
                                             = factorization_frequency + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: infinite_bound_size                 &
                                             = feasibility_tolerance + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: iterations_limit                    &
                                             = infinite_bound_size + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: lu_density_tolerance                &
                                             = iterations_limit + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: lu_singularity_tolerance            &
                                             = lu_density_tolerance + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: lu_factor_tolerance                 &
                                             = lu_singularity_tolerance + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: lu_update_tolerance                 &
                                             = lu_factor_tolerance + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: lu_pivoting = lu_update_tolerance + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: minimize = lu_pivoting + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: feasible_point = minimize + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: new_basis_file = feasible_point + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: backup_basis_file                   &
                                             = new_basis_file + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: save_frequency                      &
                                             = backup_basis_file + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: list = save_frequency + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: old_basis_file = list + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: optimality_tolerance                &
                                             = old_basis_file + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: partial_price                       &
                                             = optimality_tolerance + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: pivot_tolerance = partial_price + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: print_file = pivot_tolerance + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: print_frequency = print_file + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: qpsolver = print_frequency + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: reduced_hessian_dimension           &
                                             = qpsolver + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: scale_option                        &
                                             = reduced_hessian_dimension + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: scale_tolerance = scale_option + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: scale_print = scale_tolerance + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: solution = scale_print + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: solution_file = solution + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: summary_file = solution_file + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: summary_frequency = summary_file + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: superbasics_limit                   &
                                             = summary_frequency + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: suppress_parameters                 &
                                             = superbasics_limit + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: system_information                  &
                                             = suppress_parameters + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: timing_level = system_information + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: unbounded_step_size                 &
                                             = timing_level + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: space_critical                      &
                                             = unbounded_step_size + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: deallocate_error_fatal              &
                                             = space_critical + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: prefix = deallocate_error_fatal + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: lspec = prefix
      CHARACTER( LEN = 6 ), PARAMETER :: specname = 'E04NQF'
      TYPE ( SPECFILE_item_type ), DIMENSION( lspec ) :: spec

!  Define the keywords

!  Integer key-words

      spec( error )%keyword = 'error-printout-device'
      spec( out )%keyword = 'printout-device'
      spec( print_level )%keyword = 'print-level'

      spec( check_frequency )%keyword = 'check-frequency'
      spec( crash_option )%keyword = 'crash-option'
      spec( crash_tolerance )%keyword = 'crash-tolerance'
      spec( elastic_mode )%keyword = 'elastic-mode'
      spec( elastic_objective )%keyword = 'elastic-objective'
      spec( elastic_weight )%keyword = 'elastic-weight'
      spec( expand_frequency )%keyword = 'expand-frequency'
      spec( factorization_frequency )%keyword = 'factorization-frequency'
      spec( feasibility_tolerance )%keyword = 'feasibility-tolerance'
      spec( infinite_bound_size )%keyword = 'infinite-bound-size'
      spec( iterations_limit )%keyword = 'iterations-limit'
      spec( lu_density_tolerance )%keyword = 'lu-density-tolerance'
      spec( lu_singularity_tolerance )%keyword = 'lu-singularity-tolerance'
      spec( lu_factor_tolerance )%keyword = 'lu-factor-tolerance'
      spec( lu_update_tolerance )%keyword = 'lu-update-tolerance'
      spec( lu_pivoting )%keyword = 'lu-pivoting'
      spec( minimize )%keyword = 'minimize'
      spec( feasible_point )%keyword = 'feasible-point'
      spec( new_basis_file )%keyword = 'new-basis-file'
      spec( backup_basis_file )%keyword = 'backup-basis-file'
      spec( save_frequency )%keyword = 'save-frequency'
      spec( list )%keyword = 'list'
      spec( old_basis_file )%keyword = 'old-basis-file'
      spec( optimality_tolerance )%keyword = 'optimality-tolerance'
      spec( partial_price )%keyword = 'partial-price'
      spec( pivot_tolerance )%keyword = 'pivot-tolerance'
      spec( print_file )%keyword = 'print-file'
      spec( print_frequency )%keyword = 'print-frequency'
      spec( qpsolver )%keyword = 'qpsolver'
      spec( reduced_hessian_dimension )%keyword = 'reduced-hessian-dimension'
      spec( scale_option )%keyword = 'scale-option'
      spec( scale_tolerance )%keyword = 'scale-tolerance'
      spec( scale_print )%keyword = 'scale-print'
      spec( solution )%keyword = 'solution'
      spec( solution_file )%keyword = 'solution-file'
      spec( summary_file )%keyword = 'summary-file'
      spec( summary_frequency )%keyword = 'summary-frequency'
      spec( superbasics_limit )%keyword = 'superbasics-limit'
      spec( suppress_parameters )%keyword = 'suppress_parameters'
      spec( system_information )%keyword = 'system-information'
      spec( timing_level )%keyword = 'timing-level'
      spec( unbounded_step_size )%keyword = 'unbounded-step-size'

!  Real key-words

!  Logical key-words

      spec( space_critical )%keyword = 'space-critical'
      spec( deallocate_error_fatal )%keyword = 'deallocate-error-fatal'

!  Character key-words

      spec( prefix )%keyword = 'output-line-prefix'

!     IF ( PRESENT( alt_specname ) ) WRITE(6,*) ' e04nqf: ', alt_specname

!  Read the specfile

      IF ( PRESENT( alt_specname ) ) THEN
        CALL SPECFILE_read( device, alt_specname, spec, lspec, control%error )
      ELSE
        CALL SPECFILE_read( device, specname, spec, lspec, control%error )
      END IF

!  Interpret the result

!  Set integer values

     CALL SPECFILE_assign_value( spec( error ),                                &
                                 control%error,                                &
                                 control%error )
     CALL SPECFILE_assign_value( spec( out ),                                  &
                                 control%out,                                  &
                                 control%error )
     CALL SPECFILE_assign_value( spec( print_level ),                          &
                                 control%print_level,                          &
                                 control%error )
     CALL SPECFILE_assign_value( spec( error ),                                &
                                 control%error,                                &
                                 control%error )
     CALL SPECFILE_assign_value( spec( out ),                                  &
                                 control%out,                                  &
                                 control%error )
     CALL SPECFILE_assign_value( spec( error ),                                &
                                 control%error,                                &
                                 control%error )
     CALL SPECFILE_assign_value( spec( check_frequency ),                      &
                                 control%check_frequency,                      &
                                 control%error )
     CALL SPECFILE_assign_value( spec( crash_option ),                         &
                                 control%crash_option,                         &
                                 control%error )
     CALL SPECFILE_assign_value( spec( crash_tolerance ),                      &
                                 control%crash_tolerance,                      &
                                 control%error )
     CALL SPECFILE_assign_value( spec( elastic_mode ),                         &
                                 control%elastic_mode,                         &
                                 control%error )
     CALL SPECFILE_assign_value( spec( elastic_objective ),                    &
                                 control%elastic_objective,                    &
                                 control%error )
     CALL SPECFILE_assign_value( spec( elastic_weight ),                       &
                                 control%elastic_weight,                       &
                                 control%error )
     CALL SPECFILE_assign_value( spec( expand_frequency ),                     &
                                 control%expand_frequency,                     &
                                 control%error )
     CALL SPECFILE_assign_value( spec( factorization_frequency ),              &
                                 control%factorization_frequency,              &
                                 control%error )
     CALL SPECFILE_assign_value( spec( feasibility_tolerance ),                &
                                 control%feasibility_tolerance,                &
                                 control%error )
     CALL SPECFILE_assign_value( spec( infinite_bound_size ),                  &
                                 control%infinite_bound_size,                  &
                                 control%error )
     CALL SPECFILE_assign_value( spec( iterations_limit ),                     &
                                 control%iterations_limit,                     &
                                 control%error )
     CALL SPECFILE_assign_value( spec( lu_density_tolerance ),                 &
                                 control%lu_density_tolerance,                 &
                                 control%error )
     CALL SPECFILE_assign_value( spec( lu_singularity_tolerance ),             &
                                 control%lu_singularity_tolerance,             &
                                 control%error )
     CALL SPECFILE_assign_value( spec( lu_factor_tolerance ),                  &
                                 control%lu_factor_tolerance,                  &
                                 control%error )
     CALL SPECFILE_assign_value( spec( lu_update_tolerance ),                  &
                                 control%lu_update_tolerance,                  &
                                 control%error )
     CALL SPECFILE_assign_value( spec( lu_pivoting ),                          &
                                 control%lu_pivoting,                          &
                                 control%error )
     CALL SPECFILE_assign_value( spec( minimize ),                             &
                                 control%minimize,                             &
                                 control%error )
     CALL SPECFILE_assign_value( spec( feasible_point ),                       &
                                 control%feasible_point,                       &
                                 control%error )
     CALL SPECFILE_assign_value( spec( new_basis_file ),                       &
                                 control%new_basis_file,                       &
                                 control%error )
     CALL SPECFILE_assign_value( spec( backup_basis_file ),                    &
                                 control%backup_basis_file,                    &
                                 control%error )
     CALL SPECFILE_assign_value( spec( save_frequency ),                       &
                                 control%save_frequency,                       &
                                 control%error )
     CALL SPECFILE_assign_value( spec( list ),                                 &
                                 control%list,                                 &
                                 control%error )
     CALL SPECFILE_assign_value( spec( old_basis_file ),                       &
                                 control%old_basis_file,                       &
                                 control%error )
     CALL SPECFILE_assign_value( spec( optimality_tolerance ),                 &
                                 control%optimality_tolerance,                 &
                                 control%error )
     CALL SPECFILE_assign_value( spec( partial_price ),                        &
                                 control%partial_price,                        &
                                 control%error )
     CALL SPECFILE_assign_value( spec( pivot_tolerance ),                      &
                                 control%pivot_tolerance,                      &
                                 control%error )
     CALL SPECFILE_assign_value( spec( print_file ),                           &
                                 control%print_file,                           &
                                 control%error )
     CALL SPECFILE_assign_value( spec( print_frequency ),                      &
                                 control%print_frequency,                      &
                                 control%error )
     CALL SPECFILE_assign_value( spec( qpsolver ),                             &
                                 control%qpsolver,                             &
                                 control%error )
     CALL SPECFILE_assign_value( spec( reduced_hessian_dimension ),            &
                                 control%reduced_hessian_dimension,            &
                                 control%error )
     CALL SPECFILE_assign_value( spec( scale_option ),                         &
                                 control%scale_option,                         &
                                 control%error )
     CALL SPECFILE_assign_value( spec( scale_tolerance ),                      &
                                 control%scale_tolerance,                      &
                                 control%error )
     CALL SPECFILE_assign_value( spec( scale_print ),                          &
                                 control%scale_print,                          &
                                 control%error )
     CALL SPECFILE_assign_value( spec( solution ),                             &
                                 control%solution,                             &
                                 control%error )
     CALL SPECFILE_assign_value( spec( solution_file ),                        &
                                 control%solution_file,                        &
                                 control%error )
     CALL SPECFILE_assign_value( spec( summary_file ),                         &
                                 control%summary_file,                         &
                                 control%error )
     CALL SPECFILE_assign_value( spec( summary_frequency ),                    &
                                 control%summary_frequency,                    &
                                 control%error )
     CALL SPECFILE_assign_value( spec( superbasics_limit ),                    &
                                 control%superbasics_limit,                    &
                                 control%error )
     CALL SPECFILE_assign_value( spec( suppress_parameters ),                  &
                                 control%suppress_parameters,                  &
                                 control%error )
     CALL SPECFILE_assign_value( spec( system_information ),                   &
                                 control%system_information,                   &
                                 control%error )
     CALL SPECFILE_assign_value( spec( timing_level ),                         &
                                 control%timing_level,                         &
                                 control%error )
     CALL SPECFILE_assign_value( spec( unbounded_step_size ),                  &
                                 control%unbounded_step_size,                  &
                                 control%error )

!  Set real values

!  Set logical values

     CALL SPECFILE_assign_value( spec( space_critical ),                       &
                                 control%space_critical,                       &
                                 control%error )
     CALL SPECFILE_assign_value( spec( deallocate_error_fatal ),               &
                                 control%deallocate_error_fatal,               &
                                 control%error )

!  Set character values

     CALL SPECFILE_assign_value( spec( prefix ),                               &
                                 control%prefix,                               &
                                 control%error )
      RETURN

      END SUBROUTINE E04NQF_read_specfile

!-*-*-*-*-*-*-*-   E 0 4 N Q F _ S O L V E   S U B R O U T I N E   -*-*-*-*-*-*-

      SUBROUTINE E04NQF_solve( prob, data, control, inform, spec )

!  solve the quadratic program using the E04NQF NAG package

!  A - by columns
!  H - lower triangle by rows

!  dummy arguments

      TYPE ( QPT_problem_type ), INTENT( INOUT ) :: prob
      TYPE ( E04NQF_data_type ), INTENT( INOUT ) :: data
      TYPE ( E04NQF_control_type ), INTENT( IN ) :: control
      TYPE ( E04NQF_inform_type ), INTENT( OUT ) :: inform
      CHARACTER ( LEN = * ), OPTIONAL, INTENT( IN ) :: spec

!  local variables

      INTEGER ( KIND = ip_ ) :: n, m, np1, npm, nea, neh, summary_unit, status
      INTEGER ( KIND = i4_ ) :: spec_unit, i
      INTEGER ( KIND = ip_ ), PARAMETER :: len_c_w = 600
      INTEGER ( KIND = ip_ ), PARAMETER :: len_r_w = 600
      INTEGER ( KIND = ip_ ), PARAMETER :: len_i_w = 600
      LOGICAL :: is_specfile
      LOGICAL, PARAMETER :: summary = .FALSE.
      LOGICAL, PARAMETER :: debug = .FALSE.
      CHARACTER ( LEN = 8 ) :: c_dummy( 1 )
      CHARACTER ( LEN = 80 ) :: array_name
      TYPE ( CONVERT_control_type ) :: control_convert
      TYPE ( CONVERT_inform_type ) :: inform_convert

!  transfer bound data into the format required by E04NQF

      n = prob%n ; m = prob%m ; np1 = n + 1 ; npm = n + m
      IF ( data%new_structure ) THEN
        array_name = 'e04nqf: data%B_l'
        CALL SPACE_resize_array( npm, data%B_l, inform%status,                 &
               inform%alloc_status, array_name = array_name,                   &
               bad_alloc = inform%bad_alloc )
        IF ( inform%status /= GALAHAD_ok ) RETURN

        array_name = 'e04nqf: data%B_u'
        CALL SPACE_resize_array( npm, data%B_u, inform%status,                 &
               inform%alloc_status, array_name = array_name,                   &
               bad_alloc = inform%bad_alloc )
        IF ( inform%status /= GALAHAD_ok ) RETURN
      END IF
      data%B_l( : n ) = prob%X_l( : n )
      data%B_l( np1 : npm ) = prob%C_l( : m )
      data%B_u( : n ) = prob%X_u( : n )
      data%B_u( np1 : npm ) = prob%C_u( : m )

!  manipulate vectors so that they conform to E04NQF's structures

      IF ( data%new_structure ) THEN
        array_name = 'e04nqf: data%HELAST'
        CALL SPACE_resize_array( npm, data%HELAST, inform%status,              &
               inform%alloc_status, array_name = array_name,                   &
               bad_alloc = inform%bad_alloc )
        IF ( inform%status /= GALAHAD_ok ) RETURN

        array_name = 'e04nqf: data%HS'
        CALL SPACE_resize_array( npm, data%HS, inform%status,                  &
               inform%alloc_status, array_name = array_name,                   &
               bad_alloc = inform%bad_alloc )
        IF ( inform%status /= GALAHAD_ok ) RETURN
      END IF
      data%HELAST( : npm ) = 3 ; data%HS( : npm ) = 0

      IF ( data%new_structure ) THEN
        array_name = 'e04nqf: data%X'
        CALL SPACE_resize_array( npm, data%X, inform%status,                   &
               inform%alloc_status, array_name = array_name,                   &
               bad_alloc = inform%bad_alloc )
        IF ( inform%status /= GALAHAD_ok ) RETURN

        array_name = 'e04nqf: prob%Y'
        CALL SPACE_resize_array( m, prob%Y, inform%status,                     &
               inform%alloc_status, array_name = array_name,                   &
               bad_alloc = inform%bad_alloc )
        IF ( inform%status /= GALAHAD_ok ) RETURN

        array_name = 'e04nqf: prob%Z'
        CALL SPACE_resize_array( n, prob%Z, inform%status,                     &
               inform%alloc_status, array_name = array_name,                   &
               bad_alloc = inform%bad_alloc )
        IF ( inform%status /= GALAHAD_ok ) RETURN

        array_name = 'e04nqf: data%Z'
        CALL SPACE_resize_array( npm, data%Z, inform%status,                   &
               inform%alloc_status, array_name = array_name,                   &
               bad_alloc = inform%bad_alloc )
        IF ( inform%status /= GALAHAD_ok ) RETURN

        array_name = 'e04nqf: probl%C'
        CALL SPACE_resize_array( m, prob%C, inform%status,                     &
               inform%alloc_status, array_name = array_name,                   &
               bad_alloc = inform%bad_alloc )
        IF ( inform%status /= GALAHAD_ok ) RETURN
      END IF
      data%X( : n ) = prob%X( : n )

!  if necessary, convert the input A into by sparse-column format

      data%original_a = SMT_get( prob%A%type ) == 'SPARSE_BY_COLUMNS'
      IF ( data%original_a ) THEN
        nea = prob%A%ptr( np1 ) - 1
      ELSE
        prob%A%m = m ; prob%A%n = n
        CALL CONVERT_to_sparse_column_format( prob%A, data%A, control_convert, &
                                              inform_convert )
        nea = data%A%ptr( np1 ) - 1
      END IF

!  record the sparse matrix H in E04NQF's user data structure, if necessary,
!  convert the input H into by sparse-row format

      data%original_h = SMT_get( prob%H%type ) == 'SPARSE_BY_ROWS'
      IF ( data%original_h ) THEN
        neh = prob%H%ptr( np1 ) - 1
        IF ( data%new_structure ) THEN
          array_name = 'e04nqf: data%C_user'
          CALL SPACE_resize_array( 1, data%C_user, inform%status,              &
                 inform%alloc_status, array_name = array_name,                 &
                 bad_alloc = inform%bad_alloc )
          IF ( inform%status /= GALAHAD_ok ) RETURN

          array_name = 'e04nqf: data%R_user'
          CALL SPACE_resize_array( neh, data%R_user, inform%status,            &
                 inform%alloc_status, array_name = array_name,                 &
                 bad_alloc = inform%bad_alloc )
          IF ( inform%status /= GALAHAD_ok ) RETURN

          array_name = 'e04nqf: data%I_user'
          CALL SPACE_resize_array( neh + np1, data%I_user, inform%status,      &
                 inform%alloc_status, array_name = array_name,                 &
                 bad_alloc = inform%bad_alloc )
          IF ( inform%status /= GALAHAD_ok ) RETURN

          data%R_user( : neh ) = prob%H%val( : neh )
          data%I_user( : np1 ) = prob%H%ptr( : np1 )
          data%I_user( n + 2 : neh + np1 ) = prob%H%col( : neh )
        END IF
      ELSE
        prob%H%m = n ; prob%H%n = n
        CALL CONVERT_to_sparse_symmetric_row_format( prob%H, data%H,           &
                                                     control_convert,          &
                                                     inform_convert )
!write(6,*) ' h_ptr ', data%H%ptr
!write(6,*) ' h_col ', data%H%col
!write(6,*) ' h_val ', data%H%val
        neh = data%H%ptr( np1 ) - 1
        IF ( data%new_structure ) THEN
          array_name = 'e04nqf: data%C_user'
          CALL SPACE_resize_array( 1, data%C_user, inform%status,              &
                 inform%alloc_status, array_name = array_name,                 &
                 bad_alloc = inform%bad_alloc )
          IF ( inform%status /= GALAHAD_ok ) RETURN

          array_name = 'e04nqf: data%R_user'
          CALL SPACE_resize_array( neh, data%R_user, inform%status,            &
                 inform%alloc_status, array_name = array_name,                 &
                 bad_alloc = inform%bad_alloc )
          IF ( inform%status /= GALAHAD_ok ) RETURN

          array_name = 'e04nqf: data%I_user'
          CALL SPACE_resize_array( neh + np1, data%I_user, inform%status,      &
                 inform%alloc_status, array_name = array_name,                 &
                 bad_alloc = inform%bad_alloc )
          IF ( inform%status /= GALAHAD_ok ) RETURN

          data%R_user( : neh ) = data%H%val( : neh )
          data%I_user( : np1 ) = data%H%ptr( : np1 )
          data%I_user( n + 2 : neh + np1 ) = data%H%col( : neh )
        END IF
      END IF

!  provide space for E04NPF's communication arrays

      IF ( data%new_structure ) THEN
        array_name = 'e04nqf: data%C_w'
        CALL SPACE_resize_array( len_c_w, data%C_w, inform%status,             &
               inform%alloc_status, array_name = array_name,                   &
               bad_alloc = inform%bad_alloc )
        IF ( inform%status /= GALAHAD_ok ) RETURN

        array_name = 'e04nqf: data%I_w'
        CALL SPACE_resize_array( len_i_w, data%I_w, inform%status,             &
               inform%alloc_status, array_name = array_name,                   &
               bad_alloc = inform%bad_alloc )
        IF ( inform%status /= GALAHAD_ok ) RETURN

        array_name = 'e04nqf: data%R_w'
        CALL SPACE_resize_array( len_r_w, data%R_w, inform%status,             &
               inform%alloc_status, array_name = array_name,                   &
               bad_alloc = inform%bad_alloc )
        IF ( inform%status /= GALAHAD_ok ) RETURN
      END IF

!  set up the internal structures

      inform%ifail = - 1
      CALL E04NPF( data%C_w, len_c_w, data%I_w, len_i_w, data%R_w, len_r_w,    &
                   inform%ifail )
      SELECT CASE( inform%ifail )
      CASE ( - 199 )
        WRITE( control%out,                                                    &
          "( ' call to E04NQ failed, substitute dummy package called' )" )
        inform%status = GALAHAD_unavailable_solver ; RETURN
      CASE ( - 399 )
        WRITE( control%out, "( ' call to E04NQ failed, licence key expired' )" )
        inform%status = GALAHAD_no_licence ; RETURN
      CASE ( - 999 )
        GO TO 990
      END SELECT

      IF ( debug ) THEN
        DO i = 1, len_i_w
         WRITE( 29, * ) i, data%I_w( i )
        END DO
        DO i = 1, len_r_w
          WRITE( 29, * ) i, data%R_w( i )
        END DO
        DO i = 1, len_c_w
          WRITE( 29, * ) i, data%C_w( i )
        END DO
      END IF

!  optionally, write a summary file

      IF ( summary ) THEN
        inform%ifail = - 1
        summary_unit = 25
        CALL X04ACF( summary_unit, 'e04nqf_summary.txt', 1, inform%ifail )
        IF ( inform%ifail /= 0 ) THEN
          WRITE( control%out, "( ' NAG X04ACF failed to open the summary',     &
         & ' file. Error code: ', I0 )" ) inform%ifail
          inform%status = GALAHAD_error_file ; RETURN
        END IF
      END IF

!  record the unit for errr messages

     i = control%error
     CALL X04AAF( 1, i )

!  read options file, if required and present

      IF ( PRESENT( spec ) ) THEN
        INQUIRE( FILE = spec, EXIST = is_specfile )
        IF ( is_specfile ) THEN
          OPEN( NEWUNIT = spec_unit, FILE = spec, FORM = 'FORMATTED',          &
                STATUS = 'OLD' )
          REWIND( spec_unit )
          inform%ifail = - 1
          CALL E04NRF( spec_unit, data%C_w, data%I_w, data%R_w, inform%ifail )
          IF ( inform%ifail /= 0 ) THEN
            inform%status = GALAHAD_error_file ; RETURN
          END IF
          CLOSE( spec_unit )
        END IF
      END IF

!  adjust the required output

      IF ( debug ) THEN
        DO i = 1, len_i_w
         WRITE( 30, * ) i, data%I_w( i )
        END DO
        DO i = 1, len_r_w
          WRITE( 30, * ) i, data%R_w( i )
        END DO
        DO i = 1, len_c_w
          WRITE( 30, * ) i, data%C_w( i )
        END DO
      END IF

      CALL E04NQF_transfer_control( control, data%C_w, len_c_w, data%I_w,      &
                                    len_i_w, data%R_w, len_r_w, inform%ifail )

      IF ( debug ) THEN
        DO i = 1, len_i_w
         WRITE( 31, * ) i, data%I_w( i )
        END DO
        DO i = 1, len_r_w
          WRITE( 31, * ) i, data%R_w( i )
        END DO
        DO i = 1, len_c_w
          WRITE( 31, * ) i, data%C_w( i )
        END DO
      END IF

!  CALL E04NTF( 'Print File', 6, data%C_w, data%I_w, data%R_w, inform%ifail )
!  CALL E04NTF( 'Print Level', 10, data%C_w, data%I_w, data%R_w, inform%ifail )
!  CALL E04NTF( 'Print Frequency', 1, data%C_w, data%I_w, data%R_w,inform%ifail)

!  solve the problem

      inform%ns = 0
      inform%ifail = - 1
      IF ( data%original_a ) THEN
        CALL E04NQF( 'C', E04NQ_qphx, m, n, nea, 1, n, n, 0, prob%f,           &
                     prob%p_name( 1 : 8 ), prob%A%val, prob%A%row, prob%A%ptr, &
                     data%B_l, data%B_u, prob%G, c_dummy,                      &
                     data%HELAST, data%HS, data%X, prob%Y, data%Z, inform%ns,  &
                     inform%ninf, inform%sinf, prob%q,                         &
                     data%C_w, len_c_w, data%I_w, len_i_w, data%R_w, len_r_w,  &
                     data%C_user, data%I_user, data%R_user, inform%ifail )
      ELSE
        CALL E04NQF( 'C', E04NQ_qphx, m, n, nea, 1, n, n, 0, prob%f,           &
                     prob%p_name( 1 : 8 ), data%A%val, data%A%row, data%A%ptr, &
                     data%B_l, data%B_u, prob%G, c_dummy,                      &
                     data%HELAST, data%HS, data%X, prob%Y, data%Z, inform%ns,  &
                     inform%ninf, inform%sinf, prob%q,                         &
                     data%C_w, len_c_w, data%I_w, len_i_w, data%R_w, len_r_w,  &
                     data%C_user, data%I_user, data%R_user, inform%ifail )
      END IF

!  record the solution

      SELECT CASE ( inform%ifail )
      CASE ( 0 )
        prob%X( : n ) = data%X( : n )
        prob%C( : m ) = data%X( np1 : npm )
        prob%Z( : n ) = data%Z( : n )
        inform%obj =  prob%q
        inform%status = GALAHAD_ok

!  compute the primal and dual residuals

        array_name = 'e04nqf: probl%R'
        CALL SPACE_resize_array( n, prob%R, inform%status,                     &
               inform%alloc_status, array_name = array_name,                   &
               bad_alloc = inform%bad_alloc )
        IF ( inform%status /= GALAHAD_ok ) RETURN

        CALL E04NQ_qphx( n, prob%X, prob%R, 0, data%C_user, data%I_user,       &
                         data%R_user )
        prob%R( : n ) = prob%R( : n ) + prob%G( : n ) - prob%Z( : n )
        CALL MOP_Ax( one, prob%A, prob%X, zero, prob%C, control%out,           &
                     control%error, control%print_level,                       &
                     m_matrix = m, n_matrix = n )
        CALL MOP_Ax( one, prob%A, prob%Y, - one, prob%R, control%out,          &
                     control%error, control%print_level,                       &
                     transpose = .TRUE., m_matrix = m, n_matrix = n )
        prob%C( : m ) = MIN( prob%C_u( : m ),                                  &
                             MAX( prob%C_l( : m ), prob%C( : m ) ) )           &
                          - prob%C( : m )

!  record the primal and dual infeasibility

      inform%primal_infeasibility = MAXVAL( ABS( prob%C( : m ) ) )
      inform%dual_infeasibility = MAXVAL( ABS( prob%R( : n ) ) )

      CASE ( 2, 8, 14 )
        inform%status = GALAHAD_error_restrictions
      CASE ( 3, 4 )
        inform%status = GALAHAD_error_ill_conditioned
      CASE ( 5 )
        inform%status = GALAHAD_error_primal_infeasible
      CASE ( 6 )
        inform%status = GALAHAD_error_unbounded
      CASE ( 7 )
        inform%status = GALAHAD_error_max_iterations
      CASE ( 9 )
        inform%status = GALAHAD_error_factorization
      CASE ( 10 )
        inform%status = GALAHAD_no_progress
      CASE ( 11 )
        inform%status = GALAHAD_error_inertia
      CASE ( 12, 13 )
        inform%status = GALAHAD_error_allocate
      CASE DEFAULT
        inform%status = GALAHAD_error_naglib
      END SELECT

!     WRITE( 6, * ) ' x  ', prob%X( : n )
!     CALL E04NQ_QPHX( n, prob%X, prob%X_l, 0, data%C_user, data%I_user,
!                      data%R_user )
!     WRITE( 6, * ) ' Hx ', prob%X_l( : n )

      IF ( debug ) THEN
        DO i = 1, len_i_w
         WRITE( 32, * ) i, data%I_w( i )
        END DO
        DO i = 1, len_r_w
          WRITE( 32, * ) i, data%R_w( i )
        END DO
        DO i = 1, len_c_w
          WRITE( 32, * ) i, data%C_w( i )
        END DO
      END IF

!  core optimization status metrics

!   i_w(421): total number of minor (active-set) Iterations
!   i_w(422): total number of major iterations
!   i_w(423): total number of fresh￼LU decompositions performed

     inform%iter = data%I_w( 421 )
     inform%major_iter = data%I_w( 422 )
     inform%nfacts = data%I_w( 423 )

!  memory requirements after an insufficient memory failure (alegedly)

!   i_w(121): Minimum length required for cw (mincw)
!   i_w(122): Minimum length required for iw (miniw)
!   i_w(123): Minimum length required for rw (minrw)

!  write details

      IF ( control%print_level > 0 ) THEN
        WRITE( control%out, "( /,                                              &
     &                 24('*'), ' GALAHAD QP statistics ', 24('*') //          &
     &                 ' Package used            :  E04NQF',   /,              &
     &                 ' Problem                 : ', A, /,                    &
     &                 ' # variables             = ', I0, /,                   &
     &                 ' # constraints           = ', I0, /,                   &
     &                 ' Exit code               = ', I0, /,                   &
     &                 ' Final f                 = ', G0 /,                    &
     &                 ' Minor its               = ', I0, /,                   &
     &                 ' Major its               = ', I0, /,                   &
     &                 ' Factorizations          = ', I0, /,                   &
     &                 67('*') / )" ) prob%p_name, n, m, inform%ifail, prob%q, &
                           inform%iter, inform%major_iter, inform%nfacts
!       IF ( inform%ifail == 0 ) &
!         WRITE( control%out, "(' Optimal X = ', 7F9.2 )" ) prob%X( : n )
      END IF

!  write summary data if required

      IF ( summary ) THEN
        inform%ifail = - 1
        CALL X04ADF( summary_unit, inform%ifail )
        IF ( inform%ifail /= 0 ) THEN
          WRITE( control%out, "( ' NAG X04ACF failed to close the summary',    &
         & ' file. Error code: ', I0 )" ) inform%ifail
          IF ( status /= GALAHAD_error_naglib ) inform%status = GALAHAD_error_io
          RETURN
        END IF
      END IF

!  ensure that the existing structure remains for subsequent calls until
!  QP_E04NQF_terminate removes it

      data%new_structure = .FALSE.

      RETURN

  990 CONTINUE
      inform%status = GALAHAD_error_allocate
      RETURN

!  internal subroutine

      CONTAINS

         SUBROUTINE E04NQ_qphx( ncolh, X, HX, nstate, C_user, I_user, R_user )
!        USE GALAHAD_KINDS_precision

!  given x, compute hx = H*x

!  dummy arguments

         INTEGER ( KIND = ip_ ), INTENT ( IN ) :: ncolh, nstate
         INTEGER ( KIND = ip_ ), INTENT ( INOUT ) :: I_user( * )
         REAL ( KIND = rp_ ), INTENT( IN ) :: X( ncolh )
         REAL ( KIND = rp_ ), INTENT( INOUT ) :: R_user( * )
         REAL ( KIND = rp_ ), INTENT( OUT ) :: HX( ncolh )
         CHARACTER ( len = 8 ), INTENT( INOUT ) :: C_user( * )

!  local variables

         INTEGER ( KIND = ip_ ) :: i, j, l, n_row

!  initialize

         n_row = ncolh + 1
         HX = 0.0_rp_

!  loop over the columns of H, remembering that only one triangle of H is stored

         DO j = 1, ncolh
!          write(6,*) ' col, start, end ', j,  I_user( j ), I_user( j + 1 ) - 1
           DO l = I_user( j ), I_user( j + 1 ) - 1
             i = I_user( n_row + l )
!            write(6,*) ' row ', i
             HX( i ) = HX( i ) + R_user( l ) * X( j )
             IF ( i /= j ) HX( j ) = HX( j ) + R_user( l ) * X( i )
           END DO
         END DO

         RETURN

!  end of subroutine E04NQ_qphx

        END SUBROUTINE E04NQ_qphx

!  end of E04NQF_solve

      END SUBROUTINE E04NQF_solve

!-*-*-*-*-   E 0 4 N Q F _ T E R M I N A T E   S U B R O U T I N E   -*-*-*-*-

      SUBROUTINE E04NQF_terminate( data, control, inform )

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!      ..............................................
!      .                                            .
!      .  Deallocate internal arrays at the end     .
!      .  of the computation                        .
!      .                                            .
!      ..............................................

!  Arguments:
!
!   data    see Subroutine E04NQF_initialize
!   control see Subroutine E04NQF_initialize
!   inform  see Subroutine E04NQF_solve

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!  clean up after the E04NQF solve

!  dummy arguments

      TYPE ( E04NQF_data_type ), INTENT( INOUT ) :: data
      TYPE ( E04NQF_control_type ), INTENT( IN ) :: control
      TYPE ( E04NQF_inform_type ), INTENT( INOUT ) :: inform

!  local variable

      CHARACTER ( LEN = 80 ) :: array_name

!  deallocate workspace

      array_name = 'e04nqf: data%B_l'
      CALL SPACE_dealloc_array( data%B_l, inform%status,                       &
             inform%alloc_status, array_name = array_name,                     &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

      array_name = 'e04nqf: data%B_u'
      CALL SPACE_dealloc_array( data%B_u, inform%status,                       &
             inform%alloc_status, array_name = array_name,                     &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

      array_name = 'e04nqf: data%HELAST'
      CALL SPACE_dealloc_array( data%HELAST, inform%status,                    &
             inform%alloc_status, array_name = array_name,                     &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

      array_name = 'e04nqf: data%HS'
      CALL SPACE_dealloc_array( data%HS, inform%status,                        &
             inform%alloc_status, array_name = array_name,                     &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

      array_name = 'e04nqf: data%X'
      CALL SPACE_dealloc_array( data%X, inform%status,                         &
             inform%alloc_status, array_name = array_name,                     &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

      array_name = 'e04nqf: data%Z'
      CALL SPACE_dealloc_array( data%Z, inform%status,                         &
             inform%alloc_status, array_name = array_name,                     &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

      array_name = 'e04nqf: data%C_w'
      CALL SPACE_dealloc_array( data%C_w, inform%status,                       &
             inform%alloc_status, array_name = array_name,                     &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

      array_name = 'e04nqf: data%I_w'
      CALL SPACE_dealloc_array( data%I_w, inform%status,                       &
             inform%alloc_status, array_name = array_name,                     &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

      array_name = 'e04nqf: data%R_w'
      CALL SPACE_dealloc_array( data%R_w, inform%status,                       &
             inform%alloc_status, array_name = array_name,                     &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

      array_name = 'e04nqf: data%C_user'
      CALL SPACE_dealloc_array( data%C_user, inform%status,                    &
             inform%alloc_status, array_name = array_name,                     &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

      array_name = 'e04nqf: data%I_user'
      CALL SPACE_dealloc_array( data%I_user, inform%status,                    &
             inform%alloc_status, array_name = array_name,                     &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

      array_name = 'e04nqf: data%R_user'
      CALL SPACE_dealloc_array( data%R_user, inform%status,                    &
             inform%alloc_status, array_name = array_name,                     &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

      IF ( .NOT. data%original_a ) THEN
        array_name = 'e04nqf: data%A%ptr'
        CALL SPACE_dealloc_array( data%A%ptr, inform%status,                   &
               inform%alloc_status, array_name = array_name,                   &
               bad_alloc = inform%bad_alloc, out = control%error )
        IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

        array_name = 'e04nqf: data%A%row'
        CALL SPACE_dealloc_array( data%A%row, inform%status,                   &
               inform%alloc_status, array_name = array_name,                   &
               bad_alloc = inform%bad_alloc, out = control%error )
        IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

        array_name = 'e04nqf: data%A%val'
        CALL SPACE_dealloc_array( data%A%val, inform%status,                   &
               inform%alloc_status, array_name = array_name,                   &
               bad_alloc = inform%bad_alloc, out = control%error )
        IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

        array_name = 'e04nqf: data%A%type'
        CALL SPACE_dealloc_array( data%A%type, inform%status,                  &
               inform%alloc_status, array_name = array_name,                   &
               bad_alloc = inform%bad_alloc, out = control%error )
        IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN
      END IF

      IF ( .NOT. data%original_h ) THEN
        array_name = 'e04nqf: data%H%ptr'
        CALL SPACE_dealloc_array( data%H%ptr, inform%status,                   &
               inform%alloc_status, array_name = array_name,                   &
               bad_alloc = inform%bad_alloc, out = control%error )
        IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

        array_name = 'e04nqf: data%H%row'
        CALL SPACE_dealloc_array( data%H%row, inform%status,                   &
               inform%alloc_status, array_name = array_name,                   &
               bad_alloc = inform%bad_alloc, out = control%error )
        IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

        array_name = 'e04nqf: data%H%val'
        CALL SPACE_dealloc_array( data%H%val, inform%status,                   &
               inform%alloc_status, array_name = array_name,                   &
               bad_alloc = inform%bad_alloc, out = control%error )
        IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

        array_name = 'e04nqf: data%H%type'
        CALL SPACE_dealloc_array( data%H%type, inform%status,                  &
               inform%alloc_status, array_name = array_name,                   &
               bad_alloc = inform%bad_alloc, out = control%error )
        IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN
      END IF

!  ensure that the structure will be re-initialised on any subsequent call

      data%new_structure = .TRUE.

      RETURN

!  end of E04NQF_terminate

      END SUBROUTINE E04NQF_terminate

! G A L A H A D - E 0 4 N Q F _ f u l l _ t e r m i n a t e  S U B R O U T I N E

      SUBROUTINE E04NQF_full_terminate( data, control, inform )

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!   Deallocate all private storage

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

      TYPE ( E04NQF_full_data_type ), INTENT( INOUT ) :: data
      TYPE ( E04NQF_control_type ), INTENT( IN ) :: control
      TYPE ( E04NQF_inform_type ), INTENT( INOUT ) :: inform

!-----------------------------------------------
!   L o c a l   V a r i a b l e s
!-----------------------------------------------

!     CHARACTER ( LEN = 80 ) :: array_name

!  deallocate workspace

      CALL E04NQF_terminate( data%e04nqf_data, control, inform )
      RETURN

!  end of subroutine E04NQF_full_terminate

      END SUBROUTINE E04NQF_full_terminate

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
