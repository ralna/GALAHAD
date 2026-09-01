! THIS VERSION: GALAHAD 5.6 - 2026-08-19 AT 14:50 GMT.

#include "galahad_modules.h"

!-*-*-*-*-*-*-*-*-*-*-  G A L A H A D _ Q P    M O D U L E  -*-*-*-*-*-*-*-*-*-

!  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
!  Principal author: Nick Gould

!  History -
!   originally released in GALAHAD Version 2.4. January 5th 2011
!   expanded to handle multiple external solvers, Version 5.6, July 28th 2026

!  For full documentation, see
!   http://galahad.rl.ac.uk/galahad-www/specs.html

   MODULE GALAHAD_QP_precision

!     ------------------------------------------------
!     |                                              |
!     | Minimize the quadratic objective function    |
!     |                                              |
!     |  1/2 x^T H x + g^T x + f                     |
!     |                                              |
!     | or linear/seprable objective function        |
!     |                                              |
!     |  1/2 || W * ( x - x^0 ) ||_2^2 + g^T x + f   |
!     |                                              |
!     | subject to the linear constraints and bounds |
!     |                                              |
!     |          c_l <= A x <= c_u                   |
!     |          x_l <=  x <= x_u                    |
!     |                                              |
!     | for some posibly indefinite Hessian H or     |
!     | (possibly zero) diagonal matrix W using a    |
!     | variety of methods. This provides a generic  |
!     | interface to all GALAHAD and many other      |
!     | external (if they are available) QP routines |
!     |                                              |
!     ------------------------------------------------

      USE GALAHAD_KINDS_precision, ONLY: i4_, ip_, rp_
      USE GALAHAD_CLOCK, ONLY: CLOCK_time
      USE GALAHAD_SYMBOLS, ONLY: ACTIVE => GALAHAD_ACTIVE,                     &
                                 GALAHAD_ok,                                   &
                                 GALAHAD_error_allocate,                       &
                                 GALAHAD_error_deallocate,                     &
                                 GALAHAD_error_restrictions,                   &
                                 GALAHAD_error_primal_infeasible,              &
                                 GALAHAD_error_factorization,                  &
                                 GALAHAD_error_unbounded,                      &
                                 GALAHAD_error_ill_conditioned,                &
                                 GALAHAD_no_progress,                          &
                                 GALAHAD_error_tiny_step,                      &
                                 GALAHAD_error_max_iterations,                 &
                                 GALAHAD_error_inertia,                        &
                                 GALAHAD_error_scale,                          &
                                 GALAHAD_error_presolve,                       &
                                 GALAHAD_error_qpa,                            &
                                 GALAHAD_error_qpb,                            &
                                 GALAHAD_error_qpc,                            &
                                 GALAHAD_error_cqp,                            &
                                 GALAHAD_error_dqp,                            &
                                 GALAHAD_error_cdqp,                           &
                                 GALAHAD_error_unknown_solver,                 &
                                 GALAHAD_error_restrictions,                   &
                                 GALAHAD_error_file,                           &
                                 GALAHAD_error_io,                             &
                                 GALAHAD_error_naglib,                         &
                                 GALAHAD_error_osqp,                           &
                                 GALAHAD_unavailable_solver,                   &
                                 GALAHAD_no_licence 
      USE GALAHAD_SPACE_precision, ONLY: SPACE_resize_array, SPACE_dealloc_array
      USE GALAHAD_SPECFILE_precision, ONLY: SPECFILE_item_type, SPECFILE_read, &
                                            SPECFILE_assign_value
      USE GALAHAD_SMT_precision, ONLY: SMT_TYPE, SMT_GET, SMT_PUT
      USE GALAHAD_QPT_precision, ONLY: QPT_problem_type, QPT_keyword_A,        &
                                       QPT_keyword_H, QPT_summarize_problem
      USE GALAHAD_QPD_precision, ONLY: QPD_data_type, QPD_SIF
      USE GALAHAD_SORT_precision, ONLY: SORT_reorder_by_rows
      USE GALAHAD_SCALE_precision, ONLY: SCALE_control_type,                   &
                                         SCALE_inform_type,                    &
                                         SCALE_initialize,                     &
                                         SCALE_read_specfile,                  &
                                         SCALE_get, SCALE_apply,               &
                                         SCALE_recover, SCALE_terminate
      USE GALAHAD_PRESOLVE_precision, ONLY: PRESOLVE_control_type,             &
                                            PRESOLVE_inform_type,              &
                                            PRESOLVE_initialize,               &
                                            PRESOLVE_read_specfile,            &
                                            PRESOLVE_apply, PRESOLVE_restore,  &
                                            PRESOLVE_terminate
      USE GALAHAD_MOP_precision, ONLY: mop_AX
      USE GALAHAD_CONVERT_precision
      USE GALAHAD_QPA_precision, ONLY: QPA_control_type, QPA_inform_type,      &
                                       QPA_initialize, QPA_read_specfile,      &
                                       QPA_solve, QPA_terminate
      USE GALAHAD_QPB_precision, ONLY: QPB_control_type, QPB_inform_type,      &
                                       QPB_initialize, QPB_read_specfile,      &
                                       QPB_solve, QPB_terminate
      USE GALAHAD_QPC_precision, ONLY: QPC_control_type, QPC_inform_type,      &
                                       QPC_initialize, QPC_read_specfile,      &
                                       QPC_solve, QPC_terminate
      USE GALAHAD_CQP_precision, ONLY: CQP_control_type, CQP_inform_type,      &
                                       CQP_initialize, CQP_read_specfile,      &
                                       CQP_solve, CQP_terminate
      USE GALAHAD_DQP_precision, ONLY: DQP_control_type, DQP_inform_type,      &
                                       DQP_initialize, DQP_read_specfile,      &
                                       DQP_solve, DQP_terminate
      USE GALAHAD_CDQP_precision, ONLY: CDQP_control_type, CDQP_inform_type,   &
                                        CDQP_initialize, CDQP_read_specfile,   &
                                        CDQP_solve, CDQP_terminate
      USE GALAHAD_LMS_precision, ONLY: LMS_apply_lbfgs

      USE OSQP, ONLY: OSQP_settings_type, OSQP_info_type, OSQP_data_type,      &
                      OSQP_settings, OSQP_solve, OSQP_cleanup
      USE GALAHAD_NAGLIB_precision, ONLY: E04NQF_control_type,                 &
                                          E04NQF_inform_type,                  &
                                          E04NPF, E04NQF, E04NRF, E04NTF,      &
                                          X04AAF, X04ACF, X04ADF,              &
                                          E04NQF_transfer_control

      IMPLICIT NONE ( TYPE, EXTERNAL )

      PRIVATE 
      PUBLIC :: QP_initialize, QP_read_specfile, QP_solve,                     &
                QP_terminate, QPT_problem_type, SMT_type, SMT_put, SMT_get

!----------------------
!   P a r a m e t e r s
!----------------------

      INTEGER ( KIND = ip_ ), PARAMETER :: len_solver = 20
      REAL ( KIND = rp_ ), PARAMETER :: zero = 0.0_rp_
      REAL ( KIND = rp_ ), PARAMETER :: half = 0.5_rp_
      REAL ( KIND = rp_ ), PARAMETER :: one = 1.0_rp_
      REAL ( KIND = rp_ ), PARAMETER :: two = 2.0_rp_
      REAL ( KIND = rp_ ), PARAMETER :: ten = 10.0_rp_
      REAL ( KIND = rp_ ), PARAMETER :: infinity = HUGE( one )
      REAL ( KIND = rp_ ), PARAMETER :: epsmch = EPSILON( one )

!-------------------------------------------------
!  D e r i v e d   t y p e   d e f i n i t i o n s
!-------------------------------------------------

!  - - - - - - - - - - - - - - - - - - - - - - -
!   control derived type with component defaults
!  - - - - - - - - - - - - - - - - - - - - - - -

      TYPE, PUBLIC :: QP_control_type

!   error and warning diagnostics occur on stream error

        INTEGER ( KIND = ip_ ) :: error = 6

!   general output occurs on stream out

        INTEGER ( KIND = ip_ ) :: out = 6

!   the level of output required is specified by print_level

        INTEGER ( KIND = ip_ ) :: print_level = 0

!   scaling is controled by scale. Possible values are:
!     0  no scaling
!     1  scale to try to map all variables and constraints to [0,1]
!     2  normalize rows of K = ( H A(transpose) ) using Curtis and Reid'
!                              ( A      0       )  symmetric method
!     3  normalize rows & columns of A uing Curtis and Reid' unsymmetric method
!     4  normalize rows of A so that each has one-norm close to 1
!     5  normalize rows of K (cf 2) then normalize rows of A (cf 4)
!     6  normalize rows & columns of A (cf 3) then normalize rows of A (cf 4)
!     7  normalize rows & columns using Sinkhorn-Knopp equilibration
!    <0  apply -(scale = 1-7) above but before trying presolve

        INTEGER ( KIND = ip_ ) :: scale = 0

!    specifies the unit number to write generated SIF file describing the
!     current problem

        INTEGER ( KIND = ip_ ) :: sif_file_device = 52

!   any bound larger than infinity in modulus will be regarded as infinite

        REAL ( KIND = rp_ ) :: infinity = ten ** 19

!   if %presolve true, the problem will be simplified by calling GALAHAD's
!     presolve package

        LOGICAL :: presolve = .FALSE.

!   if %space_critical true, every effort will be made to use as little
!     space as possible. This may result in longer computation time

        LOGICAL :: space_critical = .FALSE.

!   if %deallocate_error_fatal is true, any array/pointer deallocation error
!     will terminate execution. Otherwise, computation will continue

        LOGICAL :: deallocate_error_fatal = .FALSE.

!   if %generate_sif_file is .true. if a SIF file describing the current
!    problem is to be generated

        LOGICAL :: generate_sif_file = .FALSE.

!  name of generated SIF file containing input problem

        CHARACTER ( LEN = 30 ) :: sif_file_name =                              &
         "QPPROB.SIF"  // REPEAT( ' ', 19 )

!  all output lines will be prefixed by %prefix(2:LEN(TRIM(%prefix))-1)
!   where %prefix contains the required string enclosed in
!   quotes, e.g. "string" or 'string'

        CHARACTER ( LEN = 30 ) :: prefix = '""                            '

!  control parameters for SCALE

        TYPE ( SCALE_control_type ) :: SCALE_control

!  control parameters for PRESOLVE

        TYPE ( PRESOLVE_control_type ) :: PRESOLVE_control

!  control parameters for QPA

        TYPE ( QPA_control_type ) :: QPA_control

!  control parameters for QPB

        TYPE ( QPB_control_type ) :: QPB_control

!  control parameters for QPC

        TYPE ( QPC_control_type ) :: QPC_control

!  control parameters for CQP

        TYPE ( CQP_control_type ) :: CQP_control

!  control parameters for DQP

        TYPE ( DQP_control_type ) :: DQP_control

!  control parameters for CDQP

        TYPE ( CDQP_control_type ) :: CDQP_control

!  control parameters for OSQP

        TYPE ( OSQP_settings_type ) :: OSQP_control

!  control parameters for E04NQF

        TYPE ( E04NQF_control_type ) :: E04NQF_control

      END TYPE QP_control_type

!  - - - - - - - - - - - - - - - - - - - - - -
!   time derived type with component defaults
!  - - - - - - - - - - - - - - - - - - - - - -

      TYPE, PUBLIC :: QP_time_type

!  the total cpu time spent in the package

        REAL ( KIND = rp_ ) :: total = 0.0

!  the cpu time spent presolving the problem

        REAL ( KIND = rp_ ) :: presolve = 0.0

!  the cpu time spent scaling the problem

        REAL ( KIND = rp_ ) :: scale = 0.0

!  the cpu time spent in the optimization

        REAL ( KIND = rp_ ) :: solve = 0.0

!  the total clock time spent in the package

        REAL ( KIND = rp_ ) :: clock_total = 0.0

!  the clock time spent presolving the problem

        REAL ( KIND = rp_ ) :: clock_presolve = 0.0

!  the clock time spent scaling the problem

        REAL ( KIND = rp_ ) :: clock_scale = 0.0

!  the clock time spent in the optimization

        REAL ( KIND = rp_ ) :: clock_solve = 0.0

      END TYPE QP_time_type

!  - - - - - - - - - - - - - - - -
!   extended inform type for OSQP
!  - - - - - - - - - - - - - - - -

      TYPE, PUBLIC :: QP_OSQP_inform_type

!  OSQP_info derived type

         TYPE ( OSQP_info_type ) :: info

!  return status

        INTEGER ( KIND = ip_ ) :: status = 0

!  the status of the last attempted allocation/deallocation

        INTEGER ( KIND = ip_ ) :: alloc_status = 0

!  the name of the array for which an allocation/deallocation error ocurred

        CHARACTER ( LEN = 80 ) :: bad_alloc = REPEAT( ' ', 80 )

      END TYPE QP_OSQP_inform_type

!  - - - - - - - - - - - - - - - - - - - - - - -
!   inform derived type with component defaults
!  - - - - - - - - - - - - - - - - - - - - - - -

      TYPE, PUBLIC :: QP_inform_type

!  return status. See QP_solve for details

        INTEGER ( KIND = ip_ ) :: status = 0

!  the status of the last attempted allocation/deallocation

        INTEGER ( KIND = ip_ ) :: alloc_status = 0

!  the name of the array for which an allocation/deallocation error ocurred

        CHARACTER ( LEN = 80 ) :: bad_alloc = REPEAT( ' ', 80 )

!  the value of the objective function at the best estimate of the solution
!   determined by QP_solve

        REAL ( KIND = rp_ ) :: obj = HUGE( one )

!  the value of the primal infeasibility

        REAL ( KIND = rp_ ) :: primal_infeasibility = HUGE( one )

!  the value of the dual infeasibility

        REAL ( KIND = rp_ ) :: dual_infeasibility = HUGE( one )

!  the value of the complementary slackness

        REAL ( KIND = rp_ ) :: complementary_slackness = HUGE( one )

!  the number of "iterations" required

        INTEGER ( KIND = ip_ ) :: iter = - 1

!  timings (see above)

        TYPE ( QP_time_type ) :: time

!  inform parameters for SCALE

        TYPE ( SCALE_inform_type ) :: SCALE_inform

!  inform parameters for PRESOLVE

        TYPE ( PRESOLVE_inform_type ) :: PRESOLVE_inform

!  inform parameters for QPA

        TYPE ( QPA_inform_type ) :: QPA_inform

!  inform parameters for QPB

        TYPE ( QPB_inform_type ) :: QPB_inform

!  inform parameters for QPC

        TYPE ( QPC_inform_type ) :: QPC_inform

!  inform parameters for CQP

        TYPE ( CQP_inform_type ) :: CQP_inform

!  inform parameters for DQP

        TYPE ( DQP_inform_type ) :: DQP_inform

!  inform parameters for CDQP

        TYPE ( CDQP_inform_type ) :: CDQP_inform

!  inform parameters for OSQP

        TYPE ( QP_OSQP_inform_type ) :: OSQP_inform

!  inform parameters for E04NQF

        TYPE ( E04NQF_inform_type ) :: E04NQF_inform

      END TYPE QP_inform_type

!  - - - - - - - - - - - - - - - - - - - -
!   extended data derived type for E04NQF
!  - - - - - - - - - - - - - - - - - - - -

      TYPE, PUBLIC :: QP_E04NQF_data_type
        INTEGER ( KIND = ip_ ), ALLOCATABLE, DIMENSION( : ) :: HELAST, HS
        INTEGER ( KIND = ip_ ), ALLOCATABLE, DIMENSION( : ) :: I_w, I_user
        REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : ) :: B_l, B_u
        REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : ) :: X, Z, R_w, R_user
        CHARACTER ( LEN = 8 ), ALLOCATABLE, DIMENSION( : )  :: C_w, C_user
        LOGICAL :: original_a, original_h
        LOGICAL :: new_structure = .TRUE.
        TYPE ( SMT_type ) :: A, H
      END TYPE QP_E04NQF_data_type

!  - - - - - - - - - - - - - - - - - - -
!   extended data derived type for OSQP
!  - - - - - - - - - - - - - - - - - - -

      TYPE, PUBLIC :: QP_OSQP_data_type
        INTEGER ( KIND = ip_ ) :: n, m, h_ne, a_ne
        INTEGER ( KIND = ip_ ), ALLOCATABLE, DIMENSION( : ) :: A_row, A_ptr
        REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : ) :: A_val
        REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : ) :: B_l, B_u, Y
        LOGICAL :: original_a, original_h
        TYPE ( SMT_type ) :: A, H
        TYPE ( OSQP_data_type ) :: OSQP_data
      END TYPE QP_OSQP_data_type

!  - - - - - - - - - - - - - - - - - - - - - - 
!   data derived type with component defaults
!  - - - - - - - - - - - - - - - - - - - - - -

      TYPE, PUBLIC :: QP_data_type
        CHARACTER ( LEN = 20 ) :: qp_solver = "qpc" // REPEAT( ' ', 17 )
!       CHARACTER ( LEN = 30 ) :: a_format = "COORDINATE" // REPEAT( ' ', 20 )
!       CHARACTER ( LEN = 30 ) :: h_format = "COORDINATE" // REPEAT( ' ', 20 )

!  name of linear solver used

        INTEGER ( KIND = ip_ ) :: len_solver = len_solver
        CHARACTER ( LEN = len_solver ) :: solver = REPEAT( ' ', len_solver )
        TYPE ( QPD_data_type ) :: QPD_data
        TYPE ( QP_OSQP_data_type ) :: QP_OSQP_data
        TYPE ( QP_E04NQF_data_type ) :: QP_E04NQF_data
      END TYPE QP_data_type

   CONTAINS

!-*-*-*-*-*-   Q P _ I N I T I A L I Z E   S U B R O U T I N E   -*-*-*-*-*

      SUBROUTINE QP_initialize( solver, data, control, inform )

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
!
!  Default control data for QP. This routine should be called before QP_solve
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

      CHARACTER ( LEN = * ), INTENT( IN ) :: solver
      TYPE ( QP_data_type ), INTENT( INOUT ) :: data
      TYPE ( QP_control_type ), INTENT( OUT ) :: control
      TYPE ( QP_inform_type ), INTENT( OUT ) :: inform

!  Set control parameters

      CALL SCALE_initialize( data%QPD_data%SCALE_data, control%SCALE_control,  &
                             inform%SCALE_inform )
      control%SCALE_control%prefix    = '" - SCALE:"                   '
      CALL PRESOLVE_initialize( control%PRESOLVE_control,                      &
                                inform%PRESOLVE_inform,                        &
                                data%QPD_data%PRESOLVE_data )
!     control%PRESOLVE_control%prefix = '" - PRESOLVE:"                '

!  initialize solver-specific controls

      data%len_solver = LEN( solver )
      data%solver( 1 : data%len_solver ) = solver( 1 : data%len_solver )
      SELECT CASE( solver( 1 : data%len_solver ) )

!  = QPA =

      CASE ( 'qpa', 'QPA' )
        CALL QPA_initialize( data%QPD_data, control%QPA_control,               &
                             inform%QPA_inform )
        control%QPA_control%prefix = '" - QPA:"                     '
      
!  = QPB =

      CASE ( 'qpb', 'QPB' )
        CALL QPB_initialize( data%QPD_data, control%QPB_control,               &
                             inform%QPB_inform )
        control%QPB_control%prefix = '" - QPB:"                     '

!  = QPC =

      CASE ( 'qpc', 'QPC' )
        CALL QPC_initialize( data%QPD_data, control%QPC_control,               &
                             inform%QPC_inform  )
        control%QPC_control%prefix = '" - QPC:"                     '

!  = CQP =

      CASE ( 'cqp', 'CQP' )
        CALL CQP_initialize( data%QPD_data, control%CQP_control,               &
                             inform%CQP_inform  )
        control%CQP_control%prefix = '" - CQP:"                     '

!  = DQP =

      CASE ( 'dqp', 'DQP' )
        CALL DQP_initialize( data%QPD_data, control%DQP_control,               &
                             inform%DQP_inform  )
        control%DQP_control%prefix = '" - DQP:"                     '

!  = CDQP =

      CASE ( 'cdqp', 'CDQP' )
        CALL CDQP_initialize( data%QPD_data, control%CDQP_control,             &
                              inform%CDQP_inform  )
        control%CQP_control%prefix = '" - CDQP:"                    '

!  == OSQP ==

      CASE ( 'osqp', 'OSQP' )
        control%OSQP_control%verbose = 0
        control%OSQP_control%linsys_solver = 1
      END SELECT

      inform%status = GALAHAD_ok

      RETURN

!  End of QP_initialize

      END SUBROUTINE QP_initialize

!-*-*-*-*-   Q P _ R E A D _ S P E C F I L E  S U B R O U T I N E   -*-*-*-

      SUBROUTINE QP_read_specfile( control, device, alt_specname )

!  Reads the content of a specification file, and performs the assignment of
!  values associated with given keywords to the corresponding control parameters

!  The defauly values as given by QP_initialize could (roughly)
!  have been set as:

! BEGIN QP SPECIFICATIONS (DEFAULT)
!  error-printout-device                             6
!  printout-device                                   6
!  print-level                                       0
!  scale-problem                                     0
!  sif-file-device                                   52
!  infinity-value                                    1.0D+19
!  pre-solve-problem                                 F
!  space-critical                                    F
!  deallocate-error-fatal                            F
!  generate-sif-file                                 F
!  sif-file-name                                     QPPROB.SIF
!  output-line-prefix                                ""
! END QP SPECIFICATIONS (DEFAULT)

!  Dummy arguments

      TYPE ( QP_control_type ), INTENT( INOUT ) :: control
      INTEGER ( KIND = ip_ ), INTENT( IN ) :: device
      CHARACTER( LEN = * ), OPTIONAL :: alt_specname


!  Programming: Nick Gould and Ph. Toint, January 2002.

!  Local variables

      INTEGER ( KIND = ip_ ), PARAMETER :: error = 1
      INTEGER ( KIND = ip_ ), PARAMETER :: out = error + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: print_level = out + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: scale = print_level + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: sif_file_device = scale + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: infinity = sif_file_device + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: presolve = infinity + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: space_critical = presolve + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: deallocate_error_fatal              &
                                             = space_critical + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: generate_sif_file                   &
                                             = deallocate_error_fatal + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: sif_file_name = generate_sif_file + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: prefix = sif_file_name + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: lspec = prefix
      CHARACTER( LEN = 4 ), PARAMETER :: specname = 'QP'
      TYPE ( SPECFILE_item_type ), DIMENSION( lspec ) :: spec
      TYPE ( PRESOLVE_inform_type ) :: PRESOLVE_inform

!  Define the keywords

!  Integer key-words

      spec( error )%keyword = 'error-printout-device'
      spec( out )%keyword = 'printout-device'
      spec( print_level )%keyword = 'print-level'
      spec( scale )%keyword = 'scale-problem'
      spec( sif_file_device )%keyword = 'sif-file-device'

!  Real key-words

      spec( infinity )%keyword = 'infinity-value'

!  Logical key-words

      spec( presolve )%keyword = 'pre-solve-problem'
      spec( space_critical )%keyword = 'space-critical'
      spec( deallocate_error_fatal )%keyword = 'deallocate-error-fatal'
      spec( generate_sif_file )%keyword = 'generate-sif-file'

!  Character key-words

      spec( sif_file_name )%keyword = 'sif-file-name'
      spec( prefix )%keyword = 'output-line-prefix'

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
     CALL SPECFILE_assign_value( spec( scale ),                                &
                                 control%scale,                                &
                                 control%error )
     CALL SPECFILE_assign_value( spec( sif_file_device ),                      &
                                 control%sif_file_device,                      &
                                 control%error )

!  Set real values

     CALL SPECFILE_assign_value( spec( infinity ),                             &
                                 control%infinity,                             &
                                 control%error )
!  Set logical values

     CALL SPECFILE_assign_value( spec( presolve ),                             &
                                 control%presolve,                             &
                                 control%error )
     CALL SPECFILE_assign_value( spec( space_critical ),                       &
                                 control%space_critical,                       &
                                 control%error )
     CALL SPECFILE_assign_value( spec( deallocate_error_fatal ),               &
                                 control%deallocate_error_fatal,               &
                                 control%error )
     CALL SPECFILE_assign_value( spec( generate_sif_file ),                    &
                                 control%generate_sif_file,                    &
                                 control%error )

!  Set character values

     CALL SPECFILE_assign_value( spec( sif_file_name ),                        &
                                 control%sif_file_name,                        &
                                 control%error )
     CALL SPECFILE_assign_value( spec( prefix ),                               &
                                 control%prefix,                               &
                                 control%error )


!  Make sure that inifinity is set consistently

      control%PRESOLVE_control%infinity = control%infinity
      control%QPA_control%infinity = control%infinity
      control%QPB_control%infinity = control%infinity
      control%QPC_control%infinity = control%infinity
      control%CQP_control%infinity = control%infinity
      control%DQP_control%infinity = control%infinity
      control%CDQP_control%infinity = control%infinity

!  Read the specfiles for PRESOLVE, QPA, QPB, QPC, CQP and DQP

      IF ( PRESENT( alt_specname ) ) THEN
        CALL SCALE_read_specfile( control%SCALE_control, device,               &
                              alt_specname = TRIM( alt_specname ) // '-SCALE' )
        CALL PRESOLVE_read_specfile( device, control%PRESOLVE_control,         &
                                     PRESOLVE_inform, alt_specname =           &
                                     TRIM( alt_specname ) // '-PRESOLVE' )
        CALL QPA_read_specfile( control%QPA_control, device,                   &
                                alt_specname = TRIM( alt_specname ) // '-QPA' )
        CALL QPB_read_specfile( control%QPB_control, device,                   &
                                alt_specname = TRIM( alt_specname ) // '-QPB' )
        CALL QPC_read_specfile( control%QPC_control, device,                   &
                                alt_specname = TRIM( alt_specname ) // '-QPC' )
        CALL CQP_read_specfile( control%CQP_control, device,                   &
                                alt_specname = TRIM( alt_specname ) // '-CQP' )
        CALL DQP_read_specfile( control%DQP_control, device,                   &
                                alt_specname = TRIM( alt_specname ) // '-DQP' )
        CALL CDQP_read_specfile( control%CDQP_control, device,                 &
                                alt_specname = TRIM( alt_specname ) // '-CDQP' )
      ELSE
        CALL SCALE_read_specfile( control%SCALE_control, device )
        CALL PRESOLVE_read_specfile( device, control%PRESOLVE_control,         &
                                     PRESOLVE_inform )
        CALL QPA_read_specfile( control%QPA_control, device )
        CALL QPB_read_specfile( control%QPB_control, device )
        CALL QPC_read_specfile( control%QPC_control, device )
        CALL CQP_read_specfile( control%CQP_control, device )
        CALL DQP_read_specfile( control%DQP_control, device )
        CALL CDQP_read_specfile( control%CDQP_control, device )
      END IF

      RETURN

!  End of QP_read_specfile

      END SUBROUTINE QP_read_specfile

!-*-*-*-*-*-*-*-*-*-   Q P _ S O L V E  S U B R O U T I N E   -*-*-*-*-*-*-*

      SUBROUTINE QP_solve( prob, data, control, inform, C_stat, B_stat )

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
!
!  Minimize the quadratic
!
!        1/2 x^T H x + g^T x + f
!
!  or linear/separable objective
!
!        1/2 || W * ( x - x^0 ) ||_2^2 + g^T x + f
!
!  where
!
!             (c_l)_i <= (Ax)_i <= (c_u)_i , i = 1, .... , m,
!
!  and        (x_l)_i <=   x_i  <= (x_u)_i , i = 1, .... , n,
!
!  where x is a vector of n components ( x_1, .... , x_n ),
!  A is an m by n matrix, and any of the bounds (c_l)_i, (c_u)_i
!  (x_l)_i, (x_u)_i may be infinite. The subroutine is particularly
!  appropriate when H and/or A are sparse.
!
! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
!
!  Arguments:
!
!  prob is a structure of type QPT_problem_type, whose components hold
!   information about the problem on input, and its solution on output.
!   The following components must be set:
!
!   %new_problem_structure is a LOGICAL variable, which must be set to
!    .TRUE. by the user if this is the first problem with this "structure"
!    to be solved since the last call to QP_initialize, and .FALSE. if
!    a previous call to a problem with the same "structure" (but different
!    numerical data) was made.
!
!   %n is an INTEGER variable, which must be set by the user to the
!    number of optimization parameters, n.  RESTRICTION: %n >= 1
!
!   %m is an INTEGER variable, which must be set by the user to the
!    number of general linear constraints, m. RESTRICTION: %m >= 0
!
!   %Hessian_kind is an INTEGER variable which defines the type of objective
!    function to be used. Possible values are
!
!     0  all the weights will be zero, and the analytic centre of the
!        feasible region will be found. %WEIGHT (see below) need not be set
!
!     1  all the weights will be one. %WEIGHT (see below) need not be set
!
!     2  the weights will be those given by %WEIGHT (see below)
!
!    <0  the Hessian H will be used
!
!   %H is a structure of type SMT_type used to hold the LOWER TRIANGULAR part
!    of H (except for the L-BFGS case). Eight storage formats are permitted:
!
!    i) sparse, co-ordinate
!
!       In this case, the following must be set:
!
!       H%type( 1 : 10 ) = TRANSFER( 'COORDINATE', H%type )
!       H%val( : )   the values of the components of H
!       H%row( : )   the row indices of the components of H
!       H%col( : )   the column indices of the components of H
!       H%ne         the number of nonzeros used to store
!                    the LOWER TRIANGULAR part of H
!
!    ii) sparse, by rows
!
!       In this case, the following must be set:
!
!       H%type( 1 : 14 ) = TRANSFER( 'SPARSE_BY_ROWS', H%type )
!       H%val( : )   the values of the components of H, stored row by row
!       H%col( : )   the column indices of the components of H
!       H%ptr( : )   pointers to the start of each row, and past the end of
!                    the last row
!
!    iii) dense, by rows
!
!       In this case, the following must be set:
!
!       H%type( 1 : 5 ) = TRANSFER( 'DENSE', H%type )
!       H%val( : )   the values of the components of H, stored row by row,
!                    with each the entries in each row in order of
!                    increasing column indicies.
!
!    iv) diagonal
!
!       In this case, the following must be set:
!
!       H%type( 1 : 8 ) = TRANSFER( 'DIAGONAL', H%type )
!       H%val( : )   the values of the diagonals of H, stored in order
!
!   v) scaled identity
!
!       In this case, the following must be set:
!
!       H%type( 1 : 15) = 'SCALED-IDENTITY'
!       H%val( 1 )  the value assigned to each diagonal of H
!
!   vi) identity
!
!       In this case, the following must be set:
!
!       H%type( 1 : 8 ) = 'IDENTITY'
!
!   vii) no Hessian
!
!       In this case, the following must be set:
!
!       H%type( 1 : 4 ) = 'ZERO' or 'NONE'
!
!   viii) L-BFGS Hessian
!
!       In this case, the following must be set:
!
!       H%type( 1 : 5 ) = 'LBFGS'
!
!       The Hessian in this case is available via the component %H_lm below
!
!    On exit, the components will most likely have been reordered.
!    The output  matrix will be stored by rows, according to scheme (ii) above,
!    except for scheme (ix), for which a permutation will be set within H_lm.
!    However, if scheme (i) is used for input, the output H%row will contain
!    the row numbers corresponding to the values in H%val, and thus in this
!    case the output matrix will be available in both formats (i) and (ii).
!
!   %H_lm is a structure of type LMS_data_type, whose components hold the
!     L-BFGS Hessian. Access to this structure is via the module GALAHAD_LMS,
!     and this component needs only be set if %H%type( 1 : 5 ) = 'LBFGS.'
!
!   %WEIGHT is a REAL array, which need only be set if %Hessian_kind is larger
!    than 1. If this is so, it must be of length at least %n, and contain the
!    weights W for the objective function.
!
!   %target_kind is an INTEGER variable which defines possible special
!     targets X0. Possible values are
!
!     0  X0 will be a vector of zeros.
!        %X0 (see below) need not be set
!
!     1  X0 will be a vector of ones.
!        %X0 (see below) need not be set
!
!     any other value - the values of X0 will be those given by %X0 (see below)
!
!   %X0 is a REAL array, which need only be set if %Hessian_kind is larger
!    that 0 and %target_kind /= 0,1. If this is so, it must be of length at
!    least %n, and contain the targets X^0 for the objective function.
!
!   %gradient_kind is an INTEGER variable which defines the type of linear
!    term of the objective function to be used. Possible values are
!
!     0  the linear term g will be zero, and the analytic centre of the
!        feasible region will be found if in addition %Hessian_kind is 0.
!        %G (see below) need not be set
!
!     1  each component of the linear terms g will be one.
!        %G (see below) need not be set
!
!     any other value - the gradients will be those given by %G (see below)
!
!   %G is a REAL array, which need only be set if %gradient_kind is not 0
!    or 1. If this is so, it must be of length at least %n, and contain the
!    linear terms g for the objective function.
!
!   %f is a REAL variable, which must be set by the user to the value of
!    the constant term f in the objective function. On exit, it may have
!    been changed to reflect variables which have been fixed.
!
!   %A is a structure of type SMT_type used to hold the matrix A.
!    Four storage formats are permitted:
!
!    i) sparse, co-ordinate
!
!       In this case, the following must be set:
!
!       A%type( 1 : 10 ) = TRANSFER( 'COORDINATE', A%type )
!       A%val( : )   the values of the components of A
!       A%row( : )   the row indices of the components of A
!       A%col( : )   the column indices of the components of A
!       A%ne         the number of nonzeros used to store A
!
!    ii) sparse, by rows
!
!       In this case, the following must be set:
!
!       A%type( 1 : 14 ) = TRANSFER( 'SPARSE_BY_ROWS', A%type )
!       A%val( : )   the values of the components of A, stored row by row
!       A%col( : )   the column indices of the components of A
!       A%ptr( : )   pointers to the start of each row, and past the end of
!                    the last row
!
!    ii) sparse, by columns
!
!       In this case, the following must be set:
!
!       A%type( 1 : 14 ) = TRANSFER( 'SPARSE_BY_ROWS', A%type )
!       A%val( : )   the values of the components of A, stored column by column
!       A%row( : )   the row indices of the components of A
!       A%ptr( : )   pointers to the start of each colum, and past the end of
!                    the last column
!
!    iv) dense, by rows
!
!       In this case, the following must be set:
!
!       A%type( 1 : 5 ) = TRANSFER( 'DENSE', A%type )
!       A%val( : )   the values of the components of A, stored row by row,
!                    with each the entries in each row in order of
!                    increasing column indicies.
!
!    On exit, the components will most likely have been reordered.
!    The output  matrix will be stored by rows, according to scheme (ii) above.
!    However, if scheme (i) is used for input, the output A%row (resp. A%col)
!    will contain the row (resp. column) numbers corresponding to the values in 
!    A%val, and thus in this case the output matrix will be available in both 
!    formats (i) and (ii) or (i) and (iii)
!
!   %C is a REAL array of length %m, which is used to store the values of
!    A x. It need not be set on entry. On exit, it will have been filled
!    with appropriate values.
!
!   %X is a REAL array of length %n, which must be set by the user
!    to estimaes of the solution, x. On successful exit, it will contain
!    the required solution, x.
!
!   %C_l, %C_u are REAL arrays of length %n, which must be set by the user
!    to the values of the arrays c_l and c_u of lower and upper bounds on A x.
!    Any bound c_l_i or c_u_i larger than or equal to control%infinity in
!    absolute value will be regarded as being infinite (see the entry
!    control%infinity). Thus, an infinite lower bound may be specified by
!    setting the appropriate component of %C_l to a value smaller than
!    -control%infinity, while an infinite upper bound can be specified by
!    setting the appropriate element of %C_u to a value larger than
!    control%infinity. On exit, %C_l and %C_u will most likely have been
!    reordered.
!
!   %Y is a REAL array of length %m, which must be set by the user to
!    appropriate estimates of the values of the Lagrange multipliers
!    corresponding to the general constraints c_l <= A x <= c_u.
!    On successful exit, it will contain the required vector of Lagrange
!    multipliers.
!
!   %X_l, %X_u are REAL arrays of length %n, which must be set by the user
!    to the values of the arrays x_l and x_u of lower and upper bounds on x.
!    Any bound x_l_i or x_u_i larger than or equal to control%infinity in
!    absolute value will be regarded as being infinite (see the entry
!    control%infinity). Thus, an infinite lower bound may be specified by
!    setting the appropriate component of %X_l to a value smaller than
!    -control%infinity, while an infinite upper bound can be specified by
!    setting the appropriate element of %X_u to a value larger than
!    control%infinity. On exit, %X_l and %X_u will most likely have been
!    reordered.
!
!   %Z is a REAL array of length %n, which must be set by the user to
!    appropriate estimates of the values of the dual variables
!    (Lagrange multipliers corresponding to the simple bound constraints
!    x_l <= x <= x_u). On successful exit, it will contain
!   the required vector of dual variables.
!
!  data is a structure of type QP_data_type which holds private internal data
!
!  control is a structure of type QP_control_type that controls the
!   execution of the subroutine and must be set by the user. Default values for
!   the elements may be set by a call to QP_initialize. See QP_initialize
!   for details
!
!  inform is a structure of type QP_inform_type that provides
!    information on exit from QP_solve. The component status
!    has possible values:
!
!     0 Normal termination with a locally optimal solution.
!
!    -1 An allocation error occured; the status is given in the component
!       alloc_status.
!
!    -2 A deallocation error occured; the status is given in the component
!       alloc_status.
!
!   - 3 one of the restrictions
!        prob%n     >=  1
!        prob%m     >=  0
!        prob%A%type in { 'DENSE', 'SPARSE_BY_ROWS', 'COORDINATE' }
!       has been violated.
!
!    -4 The constraints are inconsistent.
!
!    -5 The constraints appear to have no feasible point.
!
!    -7 The objective function appears to be unbounded from below on the
!       feasible set.
!
!    -8 The analytic center appears to be unbounded.
!
!    -9 The analysis phase of the factorization failed; the return status
!       from the factorization package is given in the component factor_status.
!
!   -10 The factorization failed; the return status from the factorization
!       package is given in the component factor_status.
!
!   -11 The solve of a required linear system failed; the return status from
!       the factorization package is given in the component factor_status.
!
!   -16 The problem is so ill-conditoned that further progress is impossible.
!
!   -17 The step is too small to make further impact.
!
!   -18 Too many iterations have been performed.
!
!   -19 Too much elapsed CPU or system clock time has passed.
!
!  C_stat is an optional INTEGER array of length m, which if present will be
!   set on exit to indicate the likely ultimate status of the constraints.
!   Possible values are
!   C_stat( i ) < 0, the i-th constraint is likely in the active set,
!                    on its lower bound,
!               > 0, the i-th constraint is likely in the active set
!                    on its upper bound, and
!               = 0, the i-th constraint is likely not in the active set
!
!  B_stat is an optional INTEGER array of length n, which if present will be
!   set on exit to indicate the likely ultimate status of the simple bound
!   constraints. Possible values are
!   B_stat( i ) < 0, the i-th bound constraint is likely in the active set,
!                    on its lower bound,
!               > 0, the i-th bound constraint is likely in the active set
!                    on its upper bound, and
!               = 0, the i-th bound constraint is likely not in the active set
!
! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!  Dummy arguments

      TYPE ( QPT_problem_type ), INTENT( INOUT ) :: prob
      TYPE ( QP_data_type ), INTENT( INOUT ) :: data
      TYPE ( QP_control_type ), INTENT( IN ) :: control
      TYPE ( QP_inform_type ), INTENT( OUT ) :: inform
      INTEGER ( KIND = ip_ ), INTENT( OUT ), OPTIONAL,                         &
                                             DIMENSION( prob%m ) :: C_stat
      INTEGER ( KIND = ip_ ), INTENT( OUT ), OPTIONAL,                         &
                                             DIMENSION( prob%n ) :: B_stat

!  Local variables

      INTEGER ( KIND = ip_ ) :: i, scale
      REAL :: time_start, time_now, time_end
      REAL ( KIND = rp_ ) :: clock_start, clock_now, clock_end
      REAL ( KIND = rp_ ) :: val
      LOGICAL :: printi, stat_required, presolve, lbfgs
      CHARACTER ( LEN = 80 ) :: array_name
!     CHARACTER ( LEN = 30 ) :: a_format, h_format

!  prefix for all output

      CHARACTER ( LEN = LEN( TRIM( control%prefix ) ) - 2 ) :: prefix
      IF ( LEN( TRIM( control%prefix ) ) > 2 )                                 &
        prefix = control%prefix( 2 : LEN( TRIM( control%prefix ) ) - 1 )

      IF ( control%out > 0 .AND. control%print_level >= 5 )                    &
        WRITE( control%out, "( A, ' entering QP_solve ' )" ) prefix

! -------------------------------------------------------------------
!  If desired, generate a SIF file for problem passed

      IF ( control%generate_sif_file ) THEN
        CALL QPD_SIF( prob, control%sif_file_name, control%sif_file_device,    &
                      control%infinity, .TRUE. )
      END IF

!  SIF file generated
! -------------------------------------------------------------------

!  initialize time

      CALL CPU_TIME( time_start ) ; CALL CLOCK_time( clock_start )

!  initialize counts

      inform%status = GALAHAD_ok
      stat_required = PRESENT( C_stat ) .AND. PRESENT( B_stat )

!  basic single line of output per iteration

      printi = control%out > 0 .AND. control%print_level >= 1

!  ensure that input parameters are within allowed ranges

      IF ( prob%n < 1 .OR. prob%m < 0 .OR.                                     &
           .NOT. QPT_keyword_A( prob%A%type ) .OR.                             &
           .NOT. QPT_keyword_H( prob%H%type ) ) THEN
        inform%status = GALAHAD_error_restrictions
        IF ( control%error > 0 .AND. control%print_level > 0 )                 &
          WRITE( control%error,                                                &
           "( ' ', /, A, '   **  Error return ', I0, ' from QP ' )" )          &
          prefix, inform%status
        GO TO 800
      END IF

!  is an L-BFGS Hessian used?

      lbfgs = SMT_get( prob%H%type ) == 'LBFGS'

!  record matrix dimensions

      prob%A%m = prob%m
      prob%A%n = prob%n
      prob%H%m = prob%n
      prob%H%n = prob%n

      IF ( SMT_get( prob%H%type ) == 'NONE' .OR.                               &
           SMT_get( prob%H%type ) == 'ZERO' .OR.                               &
           SMT_get( prob%H%type ) == 'IDENTITY' ) THEN
        prob%H%ne = 0
      ELSE IF ( SMT_get( prob%H%type ) == 'SCALED_IDENTITY' ) THEN
        prob%H%ne = 1
      ELSE IF ( SMT_get( prob%H%type ) == 'DIAGONAL' ) THEN
        prob%H%ne = prob%n
      ELSE IF ( SMT_get( prob%H%type ) == 'DENSE' ) THEN
        prob%H%ne = prob%n * prob%n
      ELSE IF ( SMT_get( prob%H%type ) == 'SPARSE_BY_ROWS' ) THEN
        prob%H%ne = prob%H%ptr( prob%n + 1 ) - 1
      END IF
      IF ( SMT_get( prob%A%type ) == 'DENSE' ) THEN
        prob%A%ne = prob%m * prob%n
      ELSE IF ( SMT_get( prob%A%type ) == 'SPARSE_BY_ROWS' ) THEN
        prob%A%ne = prob%A%ptr( prob%m + 1 ) - 1
      END IF

!  if required, write out problem

      IF ( control%out > 0 .AND. control%print_level >= 20 )                   &
        CALL QPT_summarize_problem( control%out, prob )

!  allocate workspace

      array_name = 'qp: data%QPD_data%SH'
      CALL SPACE_resize_array( prob%n, data%QPD_data%SH, inform%status,        &
             inform%alloc_status, array_name = array_name,                     &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

!  -----------------------------
!  scale the problem if required
!  -----------------------------
!write(6,*) ' - scale ', control%scale

      IF ( lbfgs ) THEN ! to do: remove restriction
        scale = 0
      ELSE
        scale = control%scale
      END IF
      IF ( scale < 0 ) THEN
        CALL CPU_TIME( time_now ) ; CALL CLOCK_time( clock_now )
        CALL SCALE_get( prob, - control%scale,                                 &
                        data%QPD_data%SCALE_trans, data%QPD_data%SCALE_data,   &
                        control%SCALE_control, inform%SCALE_inform )
        IF ( inform%SCALE_inform%status < 0 ) THEN
          IF ( printi ) WRITE( control%out,                                    &
            "( A, '  ERROR return from SCALE (status =', I0, ')' )" )          &
               prefix, inform%SCALE_inform%status
          CALL CPU_TIME( time_end ) ; CALL CLOCK_time( clock_end )
          inform%time%scale = inform%time%scale + time_end - time_now
          inform%time%clock_scale =                                            &
            inform%time%clock_scale + clock_end - clock_now
          inform%status = GALAHAD_error_scale ; GO TO 800
        END IF
        CALL SCALE_apply( prob, data%QPD_data%SCALE_trans,                     &
                          data%QPD_data%SCALE_data,                            &
                          control%SCALE_control, inform%SCALE_inform )
        CALL CPU_TIME( time_end ) ; CALL CLOCK_time( clock_end )
        inform%time%scale = inform%time%scale + time_end - time_now
        inform%time%clock_scale =                                              &
          inform%time%clock_scale + clock_end - clock_now
        IF ( inform%SCALE_inform%status < 0 ) THEN
          IF ( printi ) WRITE( control%out,                                    &
            "( A, '  ERROR return from SCALE (status =', I0, ')' )" )          &
               prefix, inform%SCALE_inform%status
          inform%status = GALAHAD_error_scale ; GO TO 800
        END IF
      END IF

!  if the presolver is to be used, allocate sufficient space

! to do: remove LBFGS restriction
      presolve = control%presolve .AND. .NOT. lbfgs
      IF ( presolve ) THEN
        array_name = 'qp: prob%X_status'
        CALL SPACE_resize_array( prob%n, prob%X_status, inform%status,         &
               inform%alloc_status, array_name = array_name,                   &
               deallocate_error_fatal = control%deallocate_error_fatal,        &
               exact_size = control%space_critical,                            &
               bad_alloc = inform%bad_alloc, out = control%error )
        IF ( inform%status /= GALAHAD_ok ) GO TO 900

        array_name = 'qp: prob%C_status'
        CALL SPACE_resize_array( prob%m, prob%C_status, inform%status,         &
               inform%alloc_status, array_name = array_name,                   &
               deallocate_error_fatal = control%deallocate_error_fatal,        &
               exact_size = control%space_critical,                            &
               bad_alloc = inform%bad_alloc, out = control%error )
        IF ( inform%status /= GALAHAD_ok ) GO TO 900

        array_name = 'qp: prob%Z_l'
        CALL SPACE_resize_array( prob%n, prob%Z_l, inform%status,              &
               inform%alloc_status, array_name = array_name,                   &
               deallocate_error_fatal = control%deallocate_error_fatal,        &
               exact_size = control%space_critical,                            &
               bad_alloc = inform%bad_alloc, out = control%error )
        IF ( inform%status /= GALAHAD_ok ) GO TO 900

        array_name = 'qp: prob%Z_u'
        CALL SPACE_resize_array( prob%n, prob%Z_u, inform%status,              &
               inform%alloc_status, array_name = array_name,                   &
               deallocate_error_fatal = control%deallocate_error_fatal,        &
               exact_size = control%space_critical,                            &
               bad_alloc = inform%bad_alloc, out = control%error )
        IF ( inform%status /= GALAHAD_ok ) GO TO 900

        array_name = 'qp: prob%Y_l'
        CALL SPACE_resize_array( prob%m, prob%Y_l, inform%status,              &
               inform%alloc_status, array_name = array_name,                   &
               deallocate_error_fatal = control%deallocate_error_fatal,        &
               exact_size = control%space_critical,                            &
               bad_alloc = inform%bad_alloc, out = control%error )
        IF ( inform%status /= GALAHAD_ok ) GO TO 900

        array_name = 'qp: prob%Y_u'
        CALL SPACE_resize_array( prob%m, prob%Y_u, inform%status,              &
               inform%alloc_status, array_name = array_name,                   &
               deallocate_error_fatal = control%deallocate_error_fatal,        &
               exact_size = control%space_critical,                            &
               bad_alloc = inform%bad_alloc, out = control%error )
        IF ( inform%status /= GALAHAD_ok ) GO TO 900

        prob%X_status = ACTIVE
        prob%C_status = ACTIVE
        prob%Z_l( : prob%n ) = - control%infinity
        prob%Z_u( : prob%n ) =   control%infinity
        prob%Y_l( : prob%m ) = - control%infinity
        prob%Y_u( : prob%m ) =   control%infinity

!  --------------------
!  presolve if required
!  --------------------

        IF ( printi ) WRITE( control%out,                                      &
         "( A, ' dimensions prior to presolve:',                               &
       &    ' n = ', I0, ', m = ', I0, ', a_ne = ', I0, ', h_ne = ', I0 )" )   &
            prefix, prob%n, prob%m, MAX( 0, prob%A%ne ), MAX( 0, prob%H%ne )

!  overide some defaults

!        control%PRESOLVE_control%c_accuracy =                                 &
!          ten * QP_control%QPB_control%stop_p
!        control%PRESOLVE_control%z_accuracy =                                 &
!          ten * QP_control%QPB_control%stop_d

!  ensure that data will be restored after the presolve

        data%QPD_data%PRESOLVE_control = control%PRESOLVE_control
        data%QPD_data%PRESOLVE_control%get_q = .TRUE.
        data%QPD_data%PRESOLVE_control%get_f = .TRUE.
        data%QPD_data%PRESOLVE_control%get_g = .TRUE.
        data%QPD_data%PRESOLVE_control%get_H = .TRUE.
        data%QPD_data%PRESOLVE_control%get_A = .TRUE.
        data%QPD_data%PRESOLVE_control%get_x = .TRUE.
        data%QPD_data%PRESOLVE_control%get_x_bounds = .TRUE.
        data%QPD_data%PRESOLVE_control%get_z = .TRUE.
        data%QPD_data%PRESOLVE_control%get_z_bounds = .TRUE.
        data%QPD_data%PRESOLVE_control%get_c = .TRUE.
        data%QPD_data%PRESOLVE_control%get_c_bounds = .TRUE.
        data%QPD_data%PRESOLVE_control%get_y = .TRUE.
        data%QPD_data%PRESOLVE_control%get_y_bounds = .TRUE.

!  call the presolver

        CALL CPU_TIME( time_now ) ; CALL CLOCK_time( clock_now )
        CALL PRESOLVE_initialize( data%QPD_data%PRESOLVE_control,              &
                                  inform%PRESOLVE_inform,                      &
                                  data%QPD_data%PRESOLVE_data )
        IF ( inform%PRESOLVE_inform%status < 0 ) THEN
          IF ( printi ) WRITE( control%out,                                    &
            "( A, '  ERROR return from PRESOLVE (status =', I0, ')' )" )       &
               prefix, inform%PRESOLVE_inform%status
          inform%status = GALAHAD_error_presolve ; GO TO 800
        END IF
        CALL PRESOLVE_apply( prob, data%QPD_data%PRESOLVE_control,             &
                             inform%PRESOLVE_inform,                           &
                             data%QPD_data%PRESOLVE_data )
        CALL CPU_TIME( time_end ) ; CALL CLOCK_time( clock_end )
        inform%time%presolve = inform%time%presolve + time_end - time_now
        inform%time%clock_presolve =                                           &
            inform%time%clock_presolve + clock_end - clock_now
        IF ( inform%PRESOLVE_inform%status < 0 ) THEN
          IF ( printi ) WRITE( control%out,                                    &
            "( A, '  ERROR return from PRESOLVE (status =', I0, ')' )" )       &
               prefix, inform%PRESOLVE_inform%status
          inform%status = GALAHAD_error_presolve ; GO TO 800
        END IF

        IF ( SMT_get( prob%H%type ) == 'NONE' .OR.                             &
             SMT_get( prob%H%type ) == 'ZERO' .OR.                             &
             SMT_get( prob%H%type ) == 'IDENTITY' ) THEN
          prob%H%ne = 0
        ELSE IF ( SMT_get( prob%H%type ) == 'SCALED_IDENTITY' ) THEN
          prob%H%ne = 1
        ELSE IF ( SMT_get( prob%H%type ) == 'DIAGONAL' ) THEN
          prob%H%ne = prob%n
        ELSE IF ( SMT_get( prob%H%type ) == 'DENSE' ) THEN
          prob%H%ne = prob%n * prob%n
        ELSE IF ( SMT_get( prob%H%type ) == 'SPARSE_BY_ROWS' ) THEN
          prob%H%ne = prob%H%ptr( prob%n + 1 ) - 1
        END IF

        IF ( SMT_get( prob%A%type ) == 'DENSE' ) THEN
          prob%A%ne = prob%m * prob%n
        ELSE IF ( SMT_get( prob%A%type ) == 'SPARSE_BY_ROWS' ) THEN
          prob%A%ne = prob%A%ptr( prob%m + 1 ) - 1
        END IF

        IF ( printi ) WRITE( control%out,                                      &
          "( A, ' updated dimensions:  n = ', I0,                              &
       &    ', m = ', I0, ', a_ne = ', I0, ', h_ne = ', I0, /,                 &
       &    A, ' preprocessing time = ', F0.2,                                 &
       &    ', number of transformations = ', I0, / )" )                       &
            prefix, prob%n, prob%m, MAX( 0, prob%A%ne ), MAX( 0, prob%H%ne ),  &
            prefix, time_end - time_now, inform%PRESOLVE_inform%nbr_transforms
      END IF

      IF ( prob%n > 0 ) THEN

!  -----------------------------
!  scale the problem if required
!  -----------------------------

        IF ( scale > 0 ) THEN
          CALL CPU_TIME( time_now ) ; CALL CLOCK_time( clock_now )
          CALL SCALE_get( prob, control%scale,                                 &
                          data%QPD_data%SCALE_trans, data%QPD_data%SCALE_data, &
                          control%SCALE_control, inform%SCALE_inform )
          IF ( inform%SCALE_inform%status < 0 ) THEN
            IF ( printi ) WRITE( control%out,                                  &
              "( A, '  ERROR return from SCALE (status =', I0, ')' )" )        &
                 prefix, inform%SCALE_inform%status
            CALL CPU_TIME( time_end ) ; CALL CLOCK_time( clock_end )
            inform%time%scale = inform%time%scale + time_end - time_now
            inform%time%clock_scale =                                          &
              inform%time%clock_scale + clock_end - clock_now
            inform%status = GALAHAD_error_scale ; GO TO 800
          END IF
          CALL SCALE_apply( prob, data%QPD_data%SCALE_trans,                   &
                            data%QPD_data%SCALE_data,                          &
                            control%SCALE_control, inform%SCALE_inform )
          CALL CPU_TIME( time_end ) ; CALL CLOCK_time( clock_end )
          inform%time%scale = inform%time%scale + time_end - time_now
          inform%time%clock_scale =                                            &
            inform%time%clock_scale + clock_end - clock_now
          IF ( inform%SCALE_inform%status < 0 ) THEN
            IF ( printi ) WRITE( control%out,                                  &
              "( A, '  ERROR return from SCALE (status =', I0, ')' )" )        &
                 prefix, inform%SCALE_inform%status
            inform%status = GALAHAD_error_scale ; GO TO 800
          END IF
        END IF

!  ------------------
!  call the optimizer
!  ------------------

!  allocate additional workspace

        IF ( .NOT. stat_required ) THEN
          SELECT CASE( TRIM( data%solver ) )
          CASE ( 'qpa', 'QPA', 'qpc', 'QPC' )
            array_name = 'qp: data%QPD_data%C_stat'
            CALL SPACE_resize_array( prob%m, data%QPD_data%C_stat,             &
               inform%status, inform%alloc_status, array_name = array_name,    &
               deallocate_error_fatal = control%deallocate_error_fatal,        &
               exact_size = control%space_critical,                            &
               bad_alloc = inform%bad_alloc, out = control%error )
            IF ( inform%status /= GALAHAD_ok ) GO TO 900

            array_name = 'qp: data%QPD_data%B_stat'
            CALL SPACE_resize_array( prob%n, data%QPD_data%B_stat,             &
               inform%status, inform%alloc_status, array_name = array_name,    &
               deallocate_error_fatal = control%deallocate_error_fatal,        &
               exact_size = control%space_critical,                            &
               bad_alloc = inform%bad_alloc, out = control%error )
            IF ( inform%status /= GALAHAD_ok ) GO TO 900
          END SELECT
        END IF

!  apply the slected solver

        CALL CPU_TIME( time_now ) ; CALL CLOCK_time( clock_now )
        SELECT CASE( TRIM( data%solver ) )

!  == QPA ==

        CASE ( 'qpa', 'QPA' )
          IF ( printi ) WRITE( control%out,                                    &
              "( A, ' ** GALAHAD QPA solver used **' )" ) prefix
          IF ( stat_required ) THEN
            CALL QPA_solve( prob, C_stat, B_stat, data%QPD_data,               &
                            control%QPA_control, inform%QPA_inform )
          ELSE
            CALL QPA_solve( prob, data%QPD_data%C_stat, data%QPD_data%B_stat,  &
                            data%QPD_data, control%QPA_control,                &
                            inform%QPA_inform )
          END IF
          IF ( inform%QPA_inform%status /= GALAHAD_ok .AND.                    &
               inform%QPA_inform%status /= GALAHAD_error_ill_conditioned .AND. &
               inform%QPA_inform%status /= GALAHAD_error_tiny_step ) THEN
            IF ( printi ) WRITE( control%out,                                  &
              "( A, ' GALAHAD QPA solve error status = ', I0 )" ) prefix,      &
              inform%QPA_inform%status
            inform%status = GALAHAD_error_qpa ; GO TO 800
          END IF

!  == QPB ==

        CASE ( 'qpb', 'QPB' )
          IF ( printi ) WRITE( control%out,                                    &
              "( A, ' ** GALAHAD QPB solver used **' )" ) prefix
          CALL QPB_solve( prob, data%QPD_data, control%QPB_control,            &
                          inform%QPB_inform, C_stat, B_stat )
          IF ( inform%QPB_inform%status /= GALAHAD_ok .AND.                    &
               inform%QPB_inform%status /= GALAHAD_error_ill_conditioned .AND. &
               inform%QPB_inform%status /= GALAHAD_error_tiny_step ) THEN
            IF ( printi ) WRITE( control%out,                                  &
              "( A, ' GALAHAD QPB solve error status = ', I0 )" ) prefix,      &
              inform%QPB_inform%status
            inform%status = GALAHAD_error_qpb ; GO TO 800
          END IF

!  == QPC ==

        CASE ( 'qpc', 'QPC' )
          IF ( printi ) WRITE( control%out,                                    &
              "( A, ' ** GALAHAD QPC solver used **' )" ) prefix
          IF ( stat_required ) THEN
            CALL QPC_solve( prob, C_stat, B_stat, data%QPD_data,               &
                            control%QPC_control, inform%QPC_inform )
          ELSE
            CALL QPC_solve( prob, data%QPD_data%C_stat, data%QPD_data%B_stat,  &
                            data%QPD_data, control%QPC_control,                &
                            inform%QPC_inform )
          END IF
          IF ( inform%QPC_inform%status /= GALAHAD_ok .AND.                    &
               inform%QPC_inform%status /= GALAHAD_error_ill_conditioned .AND. &
               inform%QPC_inform%status /= GALAHAD_error_tiny_step ) THEN
            IF ( printi ) WRITE( control%out,                                  &
              "( A, ' GALAHAD QPC solve error status = ', I0 )" ) prefix,      &
              inform%QPC_inform%status
            inform%status = GALAHAD_error_qpc ; GO TO 800
          END IF

!  == CQP ==

        CASE ( 'cqp', 'CQP' )
          IF ( printi ) WRITE( control%out,                                    &
              "( A, ' ** GALAHAD CQP solver used **' )" ) prefix
          CALL CQP_solve( prob, data%QPD_data, control%CQP_control,            &
                          inform%CQP_inform, C_stat, B_stat )
          IF ( inform%CQP_inform%status /= GALAHAD_ok .AND.                    &
               inform%CQP_inform%status /= GALAHAD_error_ill_conditioned .AND. &
               inform%CQP_inform%status /= GALAHAD_error_tiny_step ) THEN
            IF ( printi ) WRITE( control%out,                                  &
              "( A, ' GALAHAD CQP solve error status = ', I0 )" ) prefix,      &
              inform%CQP_inform%status
            inform%status = GALAHAD_error_cqp ; GO TO 800
          END IF

!  == DQP ==

        CASE ( 'dqp', 'DQP' )
          IF ( printi ) WRITE( control%out,                                    &
              "( A, ' ** GALAHAD DQP solver used **' )" ) prefix
          CALL DQP_solve( prob, data%QPD_data, control%DQP_control,            &
                          inform%DQP_inform, C_stat, B_stat )
          IF ( inform%DQP_inform%status /= GALAHAD_ok .AND.                    &
               inform%DQP_inform%status /= GALAHAD_error_ill_conditioned .AND. &
               inform%DQP_inform%status /= GALAHAD_error_tiny_step ) THEN
            IF ( printi ) WRITE( control%out,                                  &
              "( A, ' GALAHAD DQP solve error status = ', I0 )" ) prefix,      &
              inform%DQP_inform%status
            inform%status = GALAHAD_error_dqp ; GO TO 800
          END IF

!  == CDQP ==

        CASE ( 'cdqp', 'CDQP' )
          IF ( printi ) WRITE( control%out,                                    &
              "( A, ' ** GALAHAD CDQP solver used **' )" ) prefix
          CALL CDQP_solve( prob, data%QPD_data, control%CDQP_control,          &
                           inform%CDQP_inform, C_stat, B_stat )
          IF ( inform%CDQP_inform%status /= GALAHAD_ok .AND.                   &
               inform%CDQP_inform%status /= GALAHAD_error_ill_conditioned .AND.&
               inform%CDQP_inform%status /= GALAHAD_error_tiny_step ) THEN
            IF ( printi ) WRITE( control%out,                                  &
              "( A, ' GALAHAD CDQP solve error status = ', I0 )" ) prefix,     &
              inform%CDQP_inform%status
            inform%status = GALAHAD_error_cdqp ; GO TO 800
          END IF

!  == BPMPD ==

        CASE ( 'bpmpd', 'BPMPD' )

!  == BQPD ==

        CASE ( 'bqpd', 'BQPD' )

!  == Clarabel ==

        CASE ( 'clarabel', 'CLARABEL', 'Clarabel' )

!  == E04NQF ==

        CASE ( 'e04nqf', 'E04NQF' )
          i = control%out
          CALL QP_E04NQF_solve( prob, data%QP_E04NQF_data,                     &
                                control%E04NQF_control,                        &
                                inform%E04NQF_inform,                          &
                                control%out, control%print_level )
          inform%obj = prob%q + prob%f
          inform%iter = inform%E04NQF_inform%iter
          inform%status = inform%E04NQF_inform%status
          IF ( inform%status /= GALAHAD_ok ) THEN
            IF ( printi ) WRITE( control%out, "( A,                            &
           &  ' E04NQF solve error status = ', I0 )" ) prefix, inform%status
            GO TO 800
          END IF

!  == HiGHS ==

        CASE ( 'highs', 'HIGHS', 'HiGHS' )

!  == OSQP ==

        CASE ( 'osqp', 'OSQP' )
          IF ( printi ) WRITE( control%out,                                    &
              "( A, ' ** OSQP solver used **' )" ) prefix
          CALL QP_OSQP_solve( prob, data%QP_OSQP_data, control%OSQP_control,   &
                              inform%OSQP_inform, control%out )
          inform%status = inform%OSQP_inform%status
          inform%obj = inform%OSQP_inform%info%obj_val + prob%f
!write(6,*) ' qp: inform%obj ', inform%obj
          inform%primal_infeasibility = inform%OSQP_inform%info%prim_res
          inform%dual_infeasibility = inform%OSQP_inform%info%dual_res
          inform%complementary_slackness = inform%OSQP_inform%info%duality_gap
          inform%iter = inform%OSQP_inform%info%iter
          IF ( inform%status /= GALAHAD_ok ) THEN
            IF ( printi ) WRITE( control%out, "( A,                            &
           &  ' OSQP solve error status = ', I0 )" ) prefix, inform%status
            GO TO 800
          END IF

!  == QPALM ==

        CASE ( 'qpalm', 'QPALM' )

!  == qpOASES ==

        CASE ( 'qpoases', 'QPOASES', 'qpOASES' )

!  == SCS ==

        CASE ( 'scs', 'SCS' )

!  = unavailable solver =

        CASE DEFAULT
          inform%status = GALAHAD_error_unknown_solver ; GO TO 800
        END SELECT

        inform%status = GALAHAD_ok
        CALL CPU_TIME( time_end ) ; CALL CLOCK_time( clock_end )
        inform%time%solve = inform%time%solve + time_end - time_now
        inform%time%clock_solve =                                              &
            inform%time%clock_solve + clock_end - clock_now

!  ---------------------
!  post-process the data
!  ---------------------

!  if the problem was scaled, unscale it

        IF ( scale > 0 ) THEN
          CALL CPU_TIME( time_now ) ; CALL CLOCK_time( clock_now )
          CALL SCALE_recover( prob, data%QPD_data%SCALE_trans,                 &
                              data%QPD_data%SCALE_data,                        &
                              control%SCALE_control, inform%SCALE_inform )
          CALL CPU_TIME( time_end ) ; CALL CLOCK_time( clock_end )
          inform%time%scale = inform%time%scale + time_end - time_now
          inform%time%clock_scale =                                            &
            inform%time%clock_scale + clock_end - clock_now
          IF ( inform%SCALE_inform%status < 0 ) THEN
            IF ( printi ) WRITE( control%out,                                  &
              "( A, '  ERROR return from SCALE (status =', I0, ')' )" )        &
                prefix, inform%SCALE_inform%status
            inform%status = GALAHAD_error_scale ; GO TO 800
          END IF
        END IF
      ELSE
        inform%status = GALAHAD_ok
        inform%obj = prob%f
      END IF

!  restore from presolve

      IF ( presolve ) THEN
        CALL CPU_TIME( time_now ) ; CALL CLOCK_time( clock_now )
        CALL PRESOLVE_restore( prob, data%QPD_data%PRESOLVE_control,           &
                               inform%PRESOLVE_inform,                         &
                               data%QPD_data%PRESOLVE_data )
        IF ( inform%PRESOLVE_inform%status /= 0 .AND. printi )                 &
          WRITE( control%out, " ( /, A, ' Warning: status following',          &
       &  ' PRESOLVE_restore is ', I0, / )" )                                  &
           prefix, inform%PRESOLVE_inform%status
        CALL PRESOLVE_terminate( data%QPD_data%PRESOLVE_control,               &
                                 inform%PRESOLVE_inform,                       &
                                 data%QPD_data%PRESOLVE_data )
        IF ( inform%PRESOLVE_inform%status /= 0 .AND. printi )                 &
          WRITE( control%out, " ( /, A, ' Warning: status following',          &
       &    ' PRESOLVE_terminate is ', I5, / ) " )                             &
          prefix, inform%PRESOLVE_inform%status
        CALL CPU_TIME( time_end ) ; CALL CLOCK_time( clock_end )
        inform%time%presolve = inform%time%presolve + time_end - time_now
        inform%time%clock_presolve =                                           &
          inform%time%clock_presolve + clock_end - clock_now
        IF ( printi ) WRITE( control%out,                                      &
         "( /, A, ' postprocessing time = ', F0.2)") prefix, time_end - time_now
      END IF

!  if the problem was scaled, unscale it

      IF ( scale < 0 ) THEN
        CALL CPU_TIME( time_now ) ; CALL CLOCK_time( clock_now )
        CALL SCALE_recover( prob, data%QPD_data%SCALE_trans,                   &
                            data%QPD_data%SCALE_data,                          &
                            control%SCALE_control, inform%SCALE_inform )
        CALL CPU_TIME( time_end ) ; CALL CLOCK_time( clock_end )
        inform%time%scale = inform%time%scale + time_end - time_now
        inform%time%clock_scale =                                              &
          inform%time%clock_scale + clock_end - clock_now
        IF ( inform%SCALE_inform%status < 0 ) THEN
          IF ( printi ) WRITE( control%out,                                    &
            "( A, '  ERROR return from SCALE (status =', I0, ')' )" )          &
              prefix, inform%SCALE_inform%status
          inform%status = GALAHAD_error_scale ; GO TO 800
        END IF
      END IF

!  compute the constrain residuals

      CALL mop_AX( one, prob%A, prob%X( : prob%n ), zero, prob%C( : prob%m ),  &
                   symmetric = .FALSE., transpose = .FALSE. )

!  compute maximum contraint residual and complementary slackness

      inform%primal_infeasibility = zero ; inform%complementary_slackness = zero
      DO i = 1, prob%m
        val = prob%C( i )
        inform%primal_infeasibility = MAX( inform%primal_infeasibility,        &
           MAX( zero, prob%C_l( i ) - val,  val - prob%C_u( i ) ) )
        IF ( prob%C_l( i ) > - control%infinity ) THEN
          IF ( prob%C_u( i ) < control%infinity ) THEN
            inform%complementary_slackness =                                   &
              MAX( inform%complementary_slackness,                             &
                 MIN( ABS( ( prob%C_l( i ) - val ) * prob%Y( i ) ),            &
                      ABS( ( prob%C_u( i ) - val ) * prob%Y( i ) ) ) )
          ELSE
            inform%complementary_slackness =                                   &
              MAX( inform%complementary_slackness,                             &
                   ABS( ( prob%C_l( i ) - val ) * prob%Y( i ) ) )
          END IF
        ELSE IF ( prob%C_u( i ) < control%infinity ) THEN
          inform%complementary_slackness =                                     &
           MAX( inform%complementary_slackness,                                &
                ABS( ( prob%C_u( i ) - val ) * prob%Y( i ) ) )
        END IF
      END DO

      DO i = 1, prob%n
        val = prob%X( i )
        IF ( prob%X_l( i ) > - control%infinity ) THEN
          IF ( prob%X_u( i ) < control%infinity ) THEN
            inform%complementary_slackness =                                   &
              MAX( inform%complementary_slackness,                             &
                 MIN( ABS( ( prob%X_l( i ) - val ) * prob%Z( i ) ),            &
                      ABS( ( prob%X_u( i ) - val ) * prob%Z( i ) ) ) )
          ELSE
            inform%complementary_slackness =                                   &
              MAX( inform%complementary_slackness,                             &
                   ABS( ( prob%X_l( i ) - val ) * prob%Z( i ) ) )
          END IF
        ELSE IF ( prob%X_u( i ) < control%infinity ) THEN
          inform%complementary_slackness =                                     &
            MAX( inform%complementary_slackness,                               &
                 ABS( ( prob%X_u( i ) - val ) * prob%Z( i ) ) )
        END IF
      END DO

!  compute H * x

      IF ( lbfgs ) THEN
        CALL LMS_apply_lbfgs( prob%X( : prob%n ), prob%H_lm, i,                &
                              RESULT = data%QPD_data%SH( : prob%n ) )
      ELSE
        CALL mop_AX( one, prob%H, prob%X( : prob%n ), zero,                    &
                     data%QPD_data%SH( : prob%n ), symmetric = .TRUE.,         &
                     transpose = .FALSE. )
!                    transpose = .FALSE. , print_level = 3 )
      END IF

!  compute the objective function

      inform%obj = DOT_PRODUCT( prob%X( : prob%n ), prob%G( : prob%n ) )       &
        + half * DOT_PRODUCT( prob%X( : prob%n ),                              &
                              data%QPD_data%SH( : prob%n ) ) + prob%f

!  compute the dual residual

      data%QPD_data%SH( : prob%n )                                             &
        = data%QPD_data%SH( : prob%n ) - prob%Z( : prob%n )
      CALL mop_AX( - one, prob%A, prob%Y( : prob%m ), one,                     &
                   data%QPD_data%SH( : prob%n ), symmetric = .FALSE.,          &
                   transpose = .TRUE. )

      inform%dual_infeasibility = MAXVAL( ABS( data%QPD_data%SH( : prob%n ) ) )

!  return

  800 CONTINUE
      CALL CPU_TIME( time_end ) ; CALL CLOCK_time( clock_end )
      inform%time%total = inform%time%total + time_end - time_start
      inform%time%clock_total                                                  &
        = inform%time%clock_total + clock_end - clock_start
      IF ( control%out > 0 .AND. control%print_level >= 5 )                    &
        WRITE( control%out, "( A, ' leaving QP_solve ' )" ) prefix

      RETURN

!  allocation error

  900 CONTINUE
      inform%status = GALAHAD_error_allocate
!     CALL CPU_TIME( time_now ) ; inform%time%total = time_now - time_start
      IF ( printi ) WRITE( control%out,                                        &
        "( A, ' ** Message from -QP_solve-', /,  A,                            &
       &      ' Allocation error, for ', A, /, A, ' status = ', I0 ) " )       &
        prefix, prefix, inform%bad_alloc, inform%alloc_status
      IF ( control%out > 0 .AND. control%print_level >= 5 )                    &
        WRITE( control%out, "( A, ' leaving QP_solve ' )" ) prefix

      RETURN

!  End of QP_solve

      END SUBROUTINE QP_solve

!!$!-*-*-*-*-*-*-   Q P _ B P M P D _ S O L V E   S U B R O U T I N E   -*-*-*-*-
!!$
!!$      SUBROUTINE QP_BPMPD_solve( prob, data, settings, info, out )
!!$
!!$!  solve the quadratic program using the BPMPD package
!!$
!!$!  A - by rows
!!$!  H - lower triangle by rows
!!$
!!$!  dummy arguments
!!$
!!$      TYPE ( QPT_problem_type ), INTENT( INOUT ) :: prob
!!$      TYPE ( QP_BPMPD_data_type ), INTENT( INOUT ) :: data
!!$      TYPE ( BPMPD_control_type ), INTENT( IN ) :: control
!!$      TYPE ( BPMPD_inform_type ), INTENT( OUT ) :: inform
!!$      INTEGER ( KIND = ip_ ), INTENT( IN ) :: out
!!$
!!$!  local variables
!!$
!!$      RETURN
!!$
!!$!  End of QP_BPMPD_solve
!!$
!!$      END SUBROUTINE QP_BPMPD_solve
!!$
!!$!-*-*-*-*-*-*-   Q P _ B Q P D _ S O L V E   S U B R O U T I N E   -*-*-*-*-*-
!!$
!!$      SUBROUTINE QP_BQPD_solve( prob, data, settings, info, out )
!!$
!!$!  solve the quadratic program using the BQPD package
!!$
!!$!  A - by rows (actually A -> A' in the BQPD notation)
!!$!  H - upper triangle by co-ordinates
!!$
!!$!  dummy arguments
!!$
!!$      TYPE ( QPT_problem_type ), INTENT( INOUT ) :: prob
!!$      TYPE ( QP_BQPD_data_type ), INTENT( INOUT ) :: data
!!$      TYPE ( BQPD_control_type ), INTENT( IN ) :: control
!!$      TYPE ( BQPD_inform_type ), INTENT( OUT ) :: inform
!!$      INTEGER ( KIND = ip_ ), INTENT( IN ) :: out
!!$
!!$!  local variables
!!$
!!$      RETURN
!!$
!!$!  End of QP_BQPD_solve
!!$
!!$      END SUBROUTINE QP_BQPD_solve
!!$
!!$!-*-*-*-*-   Q P _ c l a r a b e l _ S O L V E   S U B R O U T I N E   -*-*-*-*-
!!$
!!$      SUBROUTINE QP_clarabel_solve( prob, data, settings, info, out )
!!$
!!$!  solve the quadratic program using the Clarabel package
!!$
!!$!  dummy arguments
!!$
!!$      TYPE ( QPT_problem_type ), INTENT( INOUT ) :: prob
!!$      TYPE ( QP_Clarabel_data_type ), INTENT( INOUT ) :: data
!!$      TYPE ( clarabel_control_type ), INTENT( IN ) :: control
!!$      TYPE ( clarabel_inform_type ), INTENT( OUT ) :: inform
!!$      INTEGER ( KIND = ip_ ), INTENT( IN ) :: out
!!$
!!$!  local variables
!!$
!!$      RETURN
!!$
!!$!  End of QP_clarabel_solve
!!$
!!$      END SUBROUTINE QP_clarabel_solve

!-*-*-*-*-*-   Q P _ E 0 4 N Q F _ S O L V E   S U B R O U T I N E   -*-*-*-*-

      SUBROUTINE QP_E04NQF_solve( prob, data, control, inform,                 &
                                  out, print_level, spec )

!  solve the quadratic program using the E04NQF NAG package

!  A - by columns
!  H - lower triangle by rows

!  dummy arguments

      TYPE ( QPT_problem_type ), INTENT( INOUT ) :: prob
      TYPE ( QP_E04NQF_data_type ), INTENT( INOUT ) :: data
      TYPE ( E04NQF_control_type ), INTENT( IN ) :: control
      TYPE ( E04NQF_inform_type ), INTENT( OUT ) :: inform
      INTEGER ( KIND = ip_ ), INTENT( IN ) :: out, print_level
      CHARACTER ( LEN = * ), OPTIONAL, INTENT( IN ) :: spec

!  local variables

      INTEGER ( KIND = ip_ ) :: n, m,  np1, npm, nea, neh, summary_unit, status
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

        array_name = 'e04nqf: data%Z'
        CALL SPACE_resize_array( npm, data%Z, inform%status,                   &
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
        WRITE( out,                                                            &
          "( ' call to E04NQ failed, substitute dummy package called' )" )
        inform%status = GALAHAD_unavailable_solver ; RETURN
      CASE ( - 399 )
        WRITE( out, "( ' call to E04NQ failed, licence key expired' )" )
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
          WRITE( out, "( ' NAG X04ACF failed to open the summary file.',       &
         & ' Error code: ', I0 )" ) inform%ifail
          inform%status = GALAHAD_error_file ; RETURN
        END IF
      END IF

!  record the unit for errr messages

     i = control%error_file
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

      IF ( print_level > 0 ) THEN
        WRITE( out, "( /, 24('*'), ' GALAHAD QP statistics ', 24('*') //       &
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
!         WRITE( out, "(' Optimal X = ', 7F9.2 )" ) prob%X( : n )
      END IF

!  write summary data if required

      IF ( summary ) THEN
        inform%ifail = - 1
        CALL X04ADF( summary_unit, inform%ifail ) 
        IF ( inform%ifail /= 0 ) THEN
          WRITE( out, "( ' NAG X04ACF failed to close the summary file.',      &
         & ' Error code: ', I0 )" ) inform%ifail
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
         USE GALAHAD_KINDS_precision

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

!  End of QP_E04NQF_solve

      END SUBROUTINE QP_E04NQF_solve

!-*-*-*-   Q P _ E 0 4 N Q F _ T E R M I N A T E   S U B R O U T I N E   -*-*-*-

      SUBROUTINE QP_E04NQF_terminate( data, inform )

!  clean up after the E04NQF solve

!  dummy arguments

      TYPE ( QP_E04NQF_data_type ), INTENT( INOUT ) :: data
      TYPE ( E04NQF_inform_type ), INTENT( INOUT ) :: inform

!  local variable

      CHARACTER ( LEN = 80 ) :: array_name

!  deallocate workspace

      array_name = 'e04nqf: data%B_l'
      CALL SPACE_dealloc_array( data%B_l, inform%status,                       &
             inform%alloc_status, array_name = array_name,                     &
             bad_alloc = inform%bad_alloc )

      array_name = 'e04nqf: data%B_u'
      CALL SPACE_dealloc_array( data%B_u, inform%status,                       &
             inform%alloc_status, array_name = array_name,                     &
             bad_alloc = inform%bad_alloc )

      array_name = 'e04nqf: data%HELAST'
      CALL SPACE_dealloc_array( data%HELAST, inform%status,                    &
             inform%alloc_status, array_name = array_name,                     &
             bad_alloc = inform%bad_alloc )

      array_name = 'e04nqf: data%HS'
      CALL SPACE_dealloc_array( data%HS, inform%status,                        &
             inform%alloc_status, array_name = array_name,                     &
             bad_alloc = inform%bad_alloc )

      array_name = 'e04nqf: data%X'
      CALL SPACE_dealloc_array( data%X, inform%status,                         &
             inform%alloc_status, array_name = array_name,                     &
             bad_alloc = inform%bad_alloc )

      array_name = 'e04nqf: data%Z'
      CALL SPACE_dealloc_array( data%Z, inform%status,                         &
             inform%alloc_status, array_name = array_name,                     &
             bad_alloc = inform%bad_alloc )

      array_name = 'e04nqf: data%C_w'
      CALL SPACE_dealloc_array( data%C_w, inform%status,                       &
             inform%alloc_status, array_name = array_name,                     &
             bad_alloc = inform%bad_alloc )

      array_name = 'e04nqf: data%I_w'
      CALL SPACE_dealloc_array( data%I_w, inform%status,                       &
             inform%alloc_status, array_name = array_name,                     &
             bad_alloc = inform%bad_alloc )

      array_name = 'e04nqf: data%R_w'
      CALL SPACE_dealloc_array( data%R_w, inform%status,                       &
             inform%alloc_status, array_name = array_name,                     &
             bad_alloc = inform%bad_alloc )

      array_name = 'e04nqf: data%C_user'
      CALL SPACE_dealloc_array( data%C_user, inform%status,                    &
             inform%alloc_status, array_name = array_name,                     &
             bad_alloc = inform%bad_alloc )

      array_name = 'e04nqf: data%I_user'
      CALL SPACE_dealloc_array( data%I_user, inform%status,                    &
             inform%alloc_status, array_name = array_name,                     &
             bad_alloc = inform%bad_alloc )

      array_name = 'e04nqf: data%R_user'
      CALL SPACE_dealloc_array( data%R_user, inform%status,                    &
             inform%alloc_status, array_name = array_name,                     &
             bad_alloc = inform%bad_alloc )

      IF ( .NOT. data%original_a ) THEN
        array_name = 'e04nqf: data%A%ptr'
        CALL SPACE_dealloc_array( data%A%ptr, inform%status,                   &
               inform%alloc_status, array_name = array_name,                   &
               bad_alloc = inform%bad_alloc )

        array_name = 'e04nqf: data%A%row'
        CALL SPACE_dealloc_array( data%A%row, inform%status,                   &
               inform%alloc_status, array_name = array_name,                   &
               bad_alloc = inform%bad_alloc )

        array_name = 'e04nqf: data%A%val'
        CALL SPACE_dealloc_array( data%A%val, inform%status,                   &
               inform%alloc_status, array_name = array_name,                   &
               bad_alloc = inform%bad_alloc )

        array_name = 'e04nqf: data%A%type'
        CALL SPACE_dealloc_array( data%A%type, inform%status,                  &
               inform%alloc_status, array_name = array_name,                   &
               bad_alloc = inform%bad_alloc )
      END IF

      IF ( .NOT. data%original_h ) THEN
        array_name = 'e04nqf: data%H%ptr'
        CALL SPACE_dealloc_array( data%H%ptr, inform%status,                   &
               inform%alloc_status, array_name = array_name,                   &
               bad_alloc = inform%bad_alloc )

        array_name = 'e04nqf: data%H%row'
        CALL SPACE_dealloc_array( data%H%row, inform%status,                   &
               inform%alloc_status, array_name = array_name,                   &
               bad_alloc = inform%bad_alloc )

        array_name = 'e04nqf: data%H%val'
        CALL SPACE_dealloc_array( data%H%val, inform%status,                   &
               inform%alloc_status, array_name = array_name,                   &
               bad_alloc = inform%bad_alloc )

        array_name = 'e04nqf: data%H%type'
        CALL SPACE_dealloc_array( data%H%type, inform%status,                  &
               inform%alloc_status, array_name = array_name,                   &
               bad_alloc = inform%bad_alloc )
      END IF

!  ensure that the structure will be re-initialised on any subsequent call

      data%new_structure = .TRUE.

      RETURN

!  End of QP_E04NQF_terminate

      END SUBROUTINE QP_E04NQF_terminate

!-*-*-*-*-*-*-   Q P _ H i G H S _ S O L V E   S U B R O U T I N E   -*-*-*-*-

!!$      SUBROUTINE QP_HiGHS_solve( prob, data, settings, info, out )
!!$
!!$!  solve the quadratic program using the HiGHS ASM and HiPo packages
!!$
!!$!  A - by columns, zero based
!!$!  H - lower triangle by columns, zero based
!!$
!!$!  dummy arguments
!!$
!!$      TYPE ( QPT_problem_type ), INTENT( INOUT ) :: prob
!!$      TYPE ( QP_HiGHS_data_type ), INTENT( INOUT ) :: data
!!$      TYPE ( HiGHS_control_type ), INTENT( IN ) :: control
!!$      TYPE ( HiGHS_inform_type ), INTENT( OUT ) :: inform
!!$      INTEGER ( KIND = ip_ ), INTENT( IN ) :: out
!!$
!!$!  local variables
!!$
!!$      RETURN
!!$
!!$!  End of QP_HiGHS_solve
!!$
!!$      END SUBROUTINE QP_HiGHS_solve

!-*-*-*-*-*-*-   Q P _ O S Q P _ S O L V E   S U B R O U T I N E   -*-*-*-*-*-*-

      SUBROUTINE QP_OSQP_solve( prob, data, settings, inform, out )

!  solve the quadratic program using the OSQP package

!  A - by columns
!  H - upper triangle by columns (=> lower triangle by rows)

!  dummy arguments

      TYPE ( QPT_problem_type ), INTENT( INOUT ) :: prob
      TYPE ( QP_OSQP_data_type ), INTENT( INOUT ) :: data
      TYPE ( OSQP_settings_type ), INTENT( IN ) :: settings
      TYPE ( QP_OSQP_inform_type ), INTENT( OUT ) :: inform
      INTEGER ( KIND = ip_ ), INTENT( IN ) :: out

!  local variables

      INTEGER ( KIND = ip_ ) :: i, j, k, l, m, n, n_bnds, status
!     INTEGER ( KIND = ip_ ) :: a_ne, h_ne
      TYPE ( CONVERT_control_type ) :: control_convert
      TYPE ( CONVERT_inform_type ) :: inform_convert
      CHARACTER ( LEN = SIZE( inform%info%status ) ) :: info_status
      CHARACTER ( LEN = 80 ) :: array_name

!  transfer the data into OSQP's QP format

      n = prob%n ; m = prob%m
      n_bnds = COUNT( prob%X_l > -infinity .OR. prob%X_u < infinity )
      data%m = m + n_bnds
      data%a_ne = prob%a%ne + n_bnds

      array_name = 'osqp: data%A_val'
      CALL SPACE_resize_array( data%a_ne, data%A_val, inform%status,           &
             inform%alloc_status, array_name = array_name,                     &
             bad_alloc = inform%bad_alloc )
      IF ( inform%status /= GALAHAD_ok ) RETURN

      array_name = 'osqp: data%A_row'
      CALL SPACE_resize_array( data%a_ne, data%A_row, inform%status,           &
             inform%alloc_status, array_name = array_name,                     &
             bad_alloc = inform%bad_alloc )
      IF ( inform%status /= GALAHAD_ok ) RETURN

      array_name = 'osqp: data%A_ptr'
      CALL SPACE_resize_array( n + 1, data%A_ptr, inform%status,               &
             inform%alloc_status, array_name = array_name,                     &
             bad_alloc = inform%bad_alloc )
      IF ( inform%status /= GALAHAD_ok ) RETURN

      array_name = 'osqp: data%B_l'
      CALL SPACE_resize_array( data%m, data%B_l, inform%status,                &
             inform%alloc_status, array_name = array_name,                     &
             bad_alloc = inform%bad_alloc )
      IF ( inform%status /= GALAHAD_ok ) RETURN

      array_name = 'osqp: data%B_u'
      CALL SPACE_resize_array( data%m, data%B_u, inform%status,                &
             inform%alloc_status, array_name = array_name,                     &
             bad_alloc = inform%bad_alloc )
      IF ( inform%status /= GALAHAD_ok ) RETURN

      array_name = 'osqp: data%Y'
      CALL SPACE_resize_array( data%m, data%Y, inform%status,                  &
             inform%alloc_status, array_name = array_name,                     &
             bad_alloc = inform%bad_alloc )
      IF ( inform%status /= GALAHAD_ok ) RETURN

!  if necessary, convert the input A into by sparse-column format

      data%original_a = SMT_get( prob%A%type ) == 'SPARSE_BY_COLUMNS'
      IF ( .NOT. data%original_a )                                             &
        CALL CONVERT_to_sparse_column_format( prob%A, data%A, control_convert, &
                                              inform_convert )

!  reset problem constraint data (NB 1-based integer index arrays)

      data%B_l( : m ) = prob%C_l( : m ) ; data%B_u( : m ) = prob%C_u( : m )
      l = 1 ; k = m
      DO j = 1, n
        data%A_ptr( j ) = l
        IF ( data%original_a ) THEN
          DO i = prob%A%ptr( j ), prob%A%ptr( j + 1 ) - 1
            data%A_row( l ) = prob%A%row( i )
            data%A_val( l ) = prob%A%val( i )
            l = l + 1
          END DO
        ELSE
          DO i = data%A%ptr( j ), data%A%ptr( j + 1 ) - 1
            data%A_row( l ) = data%A%row( i )
            data%A_val( l ) = data%A%val( i )
            l = l + 1
          END DO
        END IF 
        IF ( prob%X_l( j ) > - infinity .OR. prob%X_u( j ) < infinity ) THEN
          k = k + 1
          data%A_row( l ) = k
          data%A_val( l ) = 1.0_rp_
          l = l + 1
          data%B_l( k ) = prob%X_l( j ) ; data%B_u( k ) = prob%X_u( j )
        END IF
      END DO
      data%A_ptr( n + 1 ) = l

!  if necessary, convert the input H into by sparse-row format

      data%original_h = SMT_get( prob%H%type ) == 'SPARSE_BY_ROWS'
      IF ( .NOT. data%original_h )                                             &
        CALL CONVERT_to_sparse_symmetric_row_format( prob%H, data%H,           &
                                                     control_convert,          &
                                                     inform_convert )
!write(6,*) ' h_ptr ', data%H%ptr
!write(6,*) ' h_col ', data%H%col
!write(6,*) ' h_val ', data%H%val

!  establish the control settings

      CALL OSQP_settings( settings, data%OSQP_data, status )
      IF ( status /= 0 ) THEN
        WRITE( out, "( ' OSQP_settings status = ', I0, ' stopping' )" ) status
        STOP
      END IF

!  solve the problem

      IF ( data%original_h ) THEN
        CALL OSQP_solve( n, data%m, prob%H%ptr, prob%H%col, prob%H%val,        &
                         prob%G, data%A_ptr, data%A_row, data%A_val,           &
                         data%B_l, data%B_u, prob%X, data%Y,                   &
                         inform%info, data%OSQP_data, status )
      ELSE
        CALL OSQP_solve( n, data%m, data%H%ptr, data%H%col, data%H%val,        &
                         prob%G, data%A_ptr, data%A_row, data%A_val,           &
                         data%B_l, data%B_u, prob%X, data%Y,                   &
                         inform%info, data%OSQP_data, status )
      END IF 

!  record the exit status

      SELECT CASE ( inform%info%status_val )
      CASE ( 1 )
        inform%status = GALAHAD_ok
      CASE ( 2 )
        inform%status = GALAHAD_no_progress
      CASE ( 3 )
        inform%status = GALAHAD_error_primal_infeasible
      CASE ( 7 )
        inform%status = GALAHAD_error_max_iterations
      CASE ( 9 )
        inform%status = GALAHAD_error_inertia
      CASE DEFAULT
        inform%status = GALAHAD_error_osqp
      END SELECT

!  recover the solution

      IF ( status == 0 ) THEN
        prob%Y( : m ) = data%Y( : m )
        k = m
        DO j = 1, n
          IF ( prob%X_l( j ) > - infinity .OR. prob%X_u( j ) < infinity ) THEN
            k = k + 1
            prob%Z( j ) = data%Y( k )
          ELSE
            prob%Z( j ) = zero
          END IF
        END DO
        prob%q = inform%info%obj_val + prob%f

!  print details, if required

        IF ( out > 0 ) THEN
          WRITE( out, "( /, ' OSQP - Fortran interface' )" )
          WRITE( out, "( ' objective function:', ES16.8 )" ) prob%q
          WRITE( out, "( ' primal & dual residuals:', 2ES16.8 )" )             &
            inform%info%prim_res, inform%info%dual_res
          WRITE( out, "( ' x:', ( 5ES16.8 ) )" ) prob%X
          WRITE( out, "( ' y:', ( 5ES16.8 ) )" ) prob%Y
          WRITE( out, "( ' z:', ( 5ES16.8 ) )" ) prob%Z
          WRITE( out, "( 1X, I0, ' iterations' ) ") inform%info%iter
          WRITE( out, "( ' status ', A , ' (status value = ', I0, ')' )" )     &
              TRIM( TRANSFER( inform%info%status, info_status ) ),             &
              inform%info%status_val
        END IF
      ELSE
        IF ( inform%info%status_val /= 1 .AND. out > 0 ) THEN
          WRITE( out, "( ' Error. Problem not solved to optimality' )" )
        END IF
      END IF

      RETURN

!  End of QP_OSQP_solve

      END SUBROUTINE QP_OSQP_solve

!-*-*-*-*-   Q P _ O S Q P _ T E R M I N A T E   S U B R O U T I N E   -*-*-*-*-

      SUBROUTINE QP_OSQP_terminate( data, inform )

!  clean up after the OSQP solve

!  dummy arguments

      TYPE ( QP_OSQP_data_type ), INTENT( INOUT ) :: data
      TYPE ( QP_OSQP_inform_type ), INTENT( INOUT ) :: inform

!  local variable

      CHARACTER ( LEN = 80 ) :: array_name

!  clean up internal osqp data

      CALL OSQP_cleanup( data%OSQP_data, inform%status )

!  deallocate arrays

      array_name = 'osqp: data%A_val'
      CALL SPACE_dealloc_array( data%A_val, inform%status,                     &
             inform%alloc_status, array_name = array_name,                     &
             bad_alloc = inform%bad_alloc )

      array_name = 'osqp: data%A_row'
      CALL SPACE_dealloc_array( data%A_row, inform%status,                     &
             inform%alloc_status, array_name = array_name,                     &
             bad_alloc = inform%bad_alloc )

      array_name = 'osqp: data%A_ptr'
      CALL SPACE_dealloc_array( data%A_ptr, inform%status,                     &
             inform%alloc_status, array_name = array_name,                     &
             bad_alloc = inform%bad_alloc )

      array_name = 'osqp: data%B_l'
      CALL SPACE_dealloc_array( data%B_l, inform%status,                       &
             inform%alloc_status, array_name = array_name,                     &
             bad_alloc = inform%bad_alloc )

      array_name = 'osqp: data%B_u'
      CALL SPACE_dealloc_array( data%B_u, inform%status,                       &
             inform%alloc_status, array_name = array_name,                     &
             bad_alloc = inform%bad_alloc )

      array_name = 'osqp: data%Y'
      CALL SPACE_dealloc_array( data%Y, inform%status,                         &
             inform%alloc_status, array_name = array_name,                     &
             bad_alloc = inform%bad_alloc )

      IF ( .NOT. data%original_a ) THEN
        array_name = 'osqp: data%A%ptr'
        CALL SPACE_dealloc_array( data%A%ptr, inform%status,                   &
               inform%alloc_status, array_name = array_name,                   &
               bad_alloc = inform%bad_alloc )

        array_name = 'osqp: data%A%row'
        CALL SPACE_dealloc_array( data%A%row, inform%status,                   &
               inform%alloc_status, array_name = array_name,                   &
               bad_alloc = inform%bad_alloc )

        array_name = 'osqp: data%A%val'
        CALL SPACE_dealloc_array( data%A%val, inform%status,                   &
               inform%alloc_status, array_name = array_name,                   &
               bad_alloc = inform%bad_alloc )

        array_name = 'osqp: data%A%type'
        CALL SPACE_dealloc_array( data%A%type, inform%status,                  &
               inform%alloc_status, array_name = array_name,                   &
               bad_alloc = inform%bad_alloc )
      END IF

      IF ( .NOT. data%original_h ) THEN
        array_name = 'osqp: data%H%ptr'
        CALL SPACE_dealloc_array( data%H%ptr, inform%status,                   &
               inform%alloc_status, array_name = array_name,                   &
               bad_alloc = inform%bad_alloc )

        array_name = 'osqp: data%H%row'
        CALL SPACE_dealloc_array( data%H%row, inform%status,                   &
               inform%alloc_status, array_name = array_name,                   &
               bad_alloc = inform%bad_alloc )

        array_name = 'osqp: data%H%val'
        CALL SPACE_dealloc_array( data%H%val, inform%status,                   &
               inform%alloc_status, array_name = array_name,                   &
               bad_alloc = inform%bad_alloc )

        array_name = 'osqp: data%H%type'
        CALL SPACE_dealloc_array( data%H%type, inform%status,                  &
               inform%alloc_status, array_name = array_name,                   &
               bad_alloc = inform%bad_alloc )
      END IF
      RETURN

!  End of QP_OSQP_terminate

      END SUBROUTINE QP_OSQP_terminate

!-*-*-*-*-*-*-   Q P _ Q P A L M _ S O L V E   S U B R O U T I N E   -*-*-*-*-*-

!!$      SUBROUTINE QP_QPALM_solve( prob, data, settings, info, out )
!!$
!!$!  solve the quadratic program using the QPALM package
!!$
!!$!  A - by columns
!!$!  H - upper triangle by columns
!!$
!!$!  dummy arguments
!!$
!!$      TYPE ( QPT_problem_type ), INTENT( INOUT ) :: prob
!!$      TYPE ( QP_OSQP_data_type ), INTENT( INOUT ) :: data
!!$      TYPE ( OSQP_control_type ), INTENT( IN ) :: control
!!$      TYPE ( OSQP_inform_type ), INTENT( OUT ) :: inform
!!$      INTEGER ( KIND = ip_ ), INTENT( IN ) :: out
!!$
!!$!  local variables
!!$
!!$      RETURN
!!$
!!$!  End of QP_qpALM_solve
!!$
!!$      END SUBROUTINE QP_qpALM_solve
!!$
!!$!-*-*-*-*-*-*-   Q P _ qpOASES _ S O L V E   S U B R O U T I N E   -*-*-*-*-*-*-
!!$
!!$      SUBROUTINE QP_qpOASES_solve( prob, data, settings, info, out )
!!$
!!$!  solve the quadratic program using the qpOASES package
!!$
!!$!  A - by columns
!!$!  H - both triangles by columns
!!$
!!$!  dummy arguments
!!$
!!$      TYPE ( QPT_problem_type ), INTENT( INOUT ) :: prob
!!$      TYPE ( QP_qpOASES_data_type ), INTENT( INOUT ) :: data
!!$      TYPE ( qpOASES_control_type ), INTENT( IN ) :: control
!!$      TYPE ( qpOASES_inform_type ), INTENT( OUT ) :: inform
!!$      INTEGER ( KIND = ip_ ), INTENT( IN ) :: out
!!$
!!$!  local variables
!!$
!!$      RETURN
!!$
!!$!  End of QP_qpOASES_solve
!!$
!!$      END SUBROUTINE QP_qpOASES_solve
!!$
!!$!-*-*-*-*-*-*-   Q P _ S C S _ S O L V E   S U B R O U T I N E   -*-*-*-*-*-*-
!!$
!!$      SUBROUTINE QP_SCS_solve( prob, data, settings, info, out )
!!$
!!$!  solve the quadratic program using the SCS package
!!$
!!$!  dummy arguments
!!$
!!$      TYPE ( QPT_problem_type ), INTENT( INOUT ) :: prob
!!$      TYPE ( QP_SCS_data_type ), INTENT( INOUT ) :: data
!!$      TYPE ( SCS_control_type ), INTENT( IN ) :: control
!!$      TYPE ( SCS_inform_type ), INTENT( OUT ) :: inform
!!$      INTEGER ( KIND = ip_ ), INTENT( IN ) :: out
!!$
!!$!  local variables
!!$
!!$      RETURN
!!$
!!$!  End of QP_SCS_solve
!!$
!!$      END SUBROUTINE QP_SCS_solve

!-*-*-*-*-*-*-   Q P _ T E R M I N A T E   S U B R O U T I N E   -*-*-*-*-*

      SUBROUTINE QP_terminate( data, control, inform )

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!      ..............................................
!      .                                            .
!      .  Deallocate internal arrays at the end     .
!      .  of the computation                        .
!      .                                            .
!      ..............................................

!  Arguments:
!
!   data    see Subroutine QP_initialize
!   control see Subroutine QP_initialize
!   inform  see Subroutine QP_solve

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!  Dummy arguments

      TYPE ( QP_data_type ), INTENT( INOUT ) :: data
      TYPE ( QP_control_type ), INTENT( IN ) :: control
      TYPE ( QP_inform_type ), INTENT( INOUT ) :: inform

!  Local variables

      CHARACTER ( LEN = 80 ) :: array_name

!  Deallocate all arrays allocated by SCALE and PRESOLVE

      CALL SCALE_terminate( data%QPD_data%SCALE_data, control%SCALE_control,   &
                            inform%SCALE_inform,                               &
                            trans = data%QPD_data%SCALE_trans )
      IF ( inform%SCALE_inform%status /= GALAHAD_ok ) THEN
        inform%status = GALAHAD_error_deallocate
        inform%alloc_status = inform%SCALE_inform%alloc_status
        IF ( control%deallocate_error_fatal ) RETURN
      END IF

      CALL PRESOLVE_terminate( data%QPD_data%PRESOLVE_control,                 &
                               inform%PRESOLVE_inform,                         &
                               data%QPD_data%PRESOLVE_data )
      IF ( inform%PRESOLVE_inform%status /= GALAHAD_ok ) THEN
        inform%status = GALAHAD_error_deallocate
!       inform%alloc_status = inform%PRESOLVE_inform%alloc_status
        IF ( control%deallocate_error_fatal ) RETURN
      END IF

!  Deallocate all arrays allocated by the supported QP solvers

      SELECT CASE( TRIM( data%solver ) )

!  == QPA ==

      CASE ( 'qpa', 'QPA' )
        CALL QPA_terminate( data%QPD_data, control%QPA_control,                &
                            inform%QPA_inform )
        IF ( inform%QPA_inform%status /= GALAHAD_ok ) THEN
          inform%status = GALAHAD_error_deallocate
          inform%alloc_status = inform%QPA_inform%alloc_status
          IF ( control%deallocate_error_fatal ) RETURN
        END IF

!  == QPB ==

      CASE ( 'qpb', 'QPB' )
        CALL QPB_terminate( data%QPD_data, control%QPB_control,                &
                            inform%QPB_inform )
        IF ( inform%QPB_inform%status /= GALAHAD_ok ) THEN
          inform%status = GALAHAD_error_deallocate
          inform%alloc_status = inform%QPB_inform%alloc_status
          IF ( control%deallocate_error_fatal ) RETURN
        END IF

!  == QPC ==

      CASE ( 'qpc', 'QPC' )
        CALL QPC_terminate( data%QPD_data, control%QPC_control,                &
                            inform%QPC_inform )
        IF ( inform%QPC_inform%status /= GALAHAD_ok ) THEN
          inform%status = GALAHAD_error_deallocate
          inform%alloc_status = inform%QPC_inform%alloc_status
          IF ( control%deallocate_error_fatal ) RETURN
        END IF

!  == CQP ==

      CASE ( 'cqp', 'CQP' )
        CALL CQP_terminate( data%QPD_data, control%CQP_control,                &
                            inform%CQP_inform )
        IF ( inform%CQP_inform%status /= GALAHAD_ok ) THEN
          inform%status = GALAHAD_error_deallocate
          inform%alloc_status = inform%CQP_inform%alloc_status
          IF ( control%deallocate_error_fatal ) RETURN
        END IF

!  == DQP ==

      CASE ( 'dqp', 'DQP' )
        CALL DQP_terminate( data%QPD_data, control%DQP_control,                &
                            inform%DQP_inform )
        IF ( inform%DQP_inform%status /= GALAHAD_ok ) THEN
          inform%status = GALAHAD_error_deallocate
          inform%alloc_status = inform%DQP_inform%alloc_status
          IF ( control%deallocate_error_fatal ) RETURN
        END IF

!  == CDQP ==

      CASE ( 'cdqp', 'CDQP' )
        CALL CDQP_terminate( data%QPD_data, control%CDQP_control,              &
                             inform%CDQP_inform )
        IF ( inform%CDQP_inform%status /= GALAHAD_ok ) THEN
          inform%status = GALAHAD_error_deallocate
          inform%alloc_status = inform%CDQP_inform%alloc_status
          IF ( control%deallocate_error_fatal ) RETURN
        END IF

!  == BPMPD ==

      CASE ( 'bpmpd', 'BPMPD' )

!  == BQPD ==

      CASE ( 'bqpd', 'BQPD' )

!  == Clarabel ==

      CASE ( 'clarabel', 'CLARABEL', 'Clarabel' )

!  == E04NQF ==

      CASE ( 'e04nqf', 'E04NQF' )
        CALL QP_E04NQF_terminate( data%QP_E04NQF_data, inform%E04NQF_inform )

!  == HiGHS ==

      CASE ( 'highs', 'HIGHS', 'HiGHS' )

!  == OSQP ==

      CASE ( 'osqp', 'OSQP' )
        CALL QP_OSQP_terminate( data%QP_OSQP_data, inform%OSQP_inform )

!  == QPALM ==

      CASE ( 'qpalm', 'QPALM' )

!  == qpOASES ==

      CASE ( 'qpoases', 'QPOASES', 'qpOASES' )

!  == SCS ==

      CASE ( 'scs', 'SCS' )

!  = unavailable solver =

      CASE DEFAULT
        inform%status = GALAHAD_error_unknown_solver ; RETURN
      END SELECT

!  Deallocate all remaing allocated arrays

      array_name = 'qp: data%SH'
      CALL SPACE_dealloc_array( data%QPD_data%SH,                              &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'qp: data%QPD_data%C_stat'
      CALL SPACE_dealloc_array( data%QPD_data%C_stat,                          &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'qp: data%QPD_data%B_stat'
      CALL SPACE_dealloc_array( data%QPD_data%B_stat,                          &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'qp: data%QPD_data%X_status'
      CALL SPACE_dealloc_array( data%QPD_data%X_status,                        &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'qp: data%QPD_data%C_status'
      CALL SPACE_dealloc_array( data%QPD_data%C_status,                        &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      RETURN

!  End of subroutine QP_terminate

      END SUBROUTINE QP_terminate

!  End of module QP

   END MODULE GALAHAD_QP_precision
