! THIS VERSION: GALAHAD 4.3 - 2023-12-29 AT 09:30 GMT.

#include "galahad_modules.h"

!-*-*-*-*-*-*-*-*-*-  G A L A H A D _ B L L S B    M O D U L E  -*-*-*-*-*-*-*-

!  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
!  Principal author: Nick Gould

!  History -
!   originally released with GALAHAD Version 4.3, December 28th 2023

!  For full documentation, see
!   http://galahad.rl.ac.uk/galahad-www/specs.html

    MODULE GALAHAD_BLLSB_precision

!      ------------------------------------------------
!     | Minimize the least-squares objective function  |
!     |                                                |
!     |  1/2 || A_o x - b ||^2 + 1/2 weight || x ||^2  |
!     |                                                |
!     | subject to the bound constraints               |
!     |                                                |
!     |             x_l <=  x <= x_u                   |
!     |                                                |
!     | using an infeasible-point primal-dual method   |
!      ------------------------------------------------

!  ** This is essentially a wrapper for GALAHAD_CLLS with m = a_ne = 0 **

      USE GALAHAD_KINDS_precision
!$    USE omp_lib
      USE GALAHAD_CLLS_precision, BLLSB_control_type => CLLS_control_type,     &
                                  BLLSB_time_type => CLLS_time_type,           &
                                  BLLSB_inform_type => CLLS_inform_type
      USE GALAHAD_CLOCK
      USE GALAHAD_SYMBOLS
      USE GALAHAD_STRING, ONLY: STRING_pleural, STRING_verb_pleural,           &
                                STRING_ies, STRING_are, STRING_ordinal
      USE GALAHAD_SPACE_precision
      USE GALAHAD_SMT_precision
      USE GALAHAD_QPT_precision
      USE GALAHAD_SPECFILE_precision
      USE GALAHAD_LSP_precision, BLLSB_dims_type => QPT_dimensions_type
      USE GALAHAD_QPD_precision, BLLSB_data_type => QPD_data_type,             &
                                 BLLSB_AX => QPD_AX,                           &
                                 BLLSB_abs_AX => QPD_abs_AX,                   &
                                 BLLSB_AoX => QPD_A_by_col_X,                  &
                                 BLLSB_abs_AoX => QPD_abs_A_by_col_X
      USE GALAHAD_ROOTS_precision
      USE GALAHAD_SORT_precision, ONLY: SORT_inverse_permute
      USE GALAHAD_FDC_precision
      USE GALAHAD_SLS_precision
      USE GALAHAD_CRO_precision
      USE GALAHAD_FIT_precision
      USE GALAHAD_CHECKPOINT_precision
      USE GALAHAD_RPD_precision, ONLY: RPD_inform_type,                        &
                                       RPD_write_qp_problem_data

      IMPLICIT NONE

      PRIVATE
      PUBLIC :: BLLSB_initialize, BLLSB_read_specfile, BLLSB_solve,            &
                BLLSB_terminate, BLLSB_control_type, BLLSB_data_type,          &
                BLLSB_time_type, BLLSB_inform_type, BLLSB_information,         &
                BLLSB_full_initialize, BLLSB_full_terminate,                   &
                BLLSB_import, BLLSB_solve_blls, BLLSB_reset_control,           &
                QPT_problem_type, SMT_type, SMT_put, SMT_get

!----------------------
!   I n t e r f a c e s
!----------------------

     INTERFACE BLLSB_initialize
       MODULE PROCEDURE BLLSB_initialize, BLLSB_full_initialize
     END INTERFACE BLLSB_initialize

     INTERFACE BLLSB_terminate
       MODULE PROCEDURE BLLSB_terminate, BLLSB_full_terminate
     END INTERFACE BLLSB_terminate

!-------------------------------------------------
!  D e r i v e d   t y p e   d e f i n i t i o n s
!-------------------------------------------------

      TYPE, PUBLIC :: BLLSB_full_data_type
        LOGICAL :: f_indexing = .TRUE.
        TYPE ( BLLSB_data_type ) :: BLLSB_data
        TYPE ( BLLSB_control_type ) :: BLLSB_control
        TYPE ( BLLSB_inform_type ) :: BLLSB_inform
        TYPE ( QPT_problem_type ) :: prob
      END TYPE BLLSB_full_data_type

   CONTAINS

!-*-*-*-*-   B L L S B _ I N I T I A L I Z E   S U B R O U T I N E   -*-*-*-*-

      SUBROUTINE BLLSB_initialize( data, control, inform )

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
!
!  Default control data for BLLSB. This routine should be called before
!  BLLSB_solve
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

      TYPE ( BLLSB_data_type ), INTENT( INOUT ) :: data
      TYPE ( BLLSB_control_type ), INTENT( OUT ) :: control
      TYPE ( BLLSB_inform_type ), INTENT( OUT ) :: inform

      CALL CLLS_initialize( data, control, inform )
      RETURN

!  End of BLLSB_initialize

      END SUBROUTINE BLLSB_initialize

! G A L A H A D - B L L S B _ F U L L _ I N I T I A L I Z E  S U B R O U T I N E

     SUBROUTINE BLLSB_full_initialize( data, control, inform )

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!   Provide default values for BLLSB controls

!   Arguments:

!   data     private internal data
!   control  a structure containing control information. See preamble
!   inform   a structure containing output information. See preamble

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     TYPE ( BLLSB_full_data_type ), INTENT( INOUT ) :: data
     TYPE ( BLLSB_control_type ), INTENT( OUT ) :: control
     TYPE ( BLLSB_inform_type ), INTENT( OUT ) :: inform

     CALL BLLSB_initialize( data%bllsb_data, control, inform )

     RETURN

!  End of subroutine BLLSB_full_initialize

     END SUBROUTINE BLLSB_full_initialize

!-*-*-*-   B L L S B _ R E A D _ S P E C F I L E  S U B R O U T I N E   -*-*-*-

      SUBROUTINE BLLSB_read_specfile( control, device, alt_specname )

!  Reads the content of a specification file, and performs the assignment of
!  values associated with given keywords to the corresponding control parameters

!  The defauly values as given by BLLSB_initialize could (roughly)
!  have been set as:

! BEGIN BLLSB SPECIFICATIONS (DEFAULT)
!  error-printout-device                             6
!  printout-device                                   6
!  print-level                                       0
!  start-print                                       -1
!  stop-print                                        -1
!  maximum-number-of-iterations                      1000
!  maximum-number-of-pcg-iterations                  1000
!  maximum-poor-iterations-before-infeasible         200
!  barrier-fixed-until-iteration                     1
!  indicator-type-used                               3
!  arc-used                                          1
!  series-order                                      5
!  restore-problem-on-output                         2
!  sif-file-device                                   52
!  qplib-file-device                                 53
!  infinity-value                                    1.0D+19
!  absolute-primal-accuracy                          1.0D-5
!  relative-primal-accuracy                          1.0D-5
!  absolute-dual-accuracy                            1.0D-5
!  relative-dual-accuracy                            1.0D-5
!  absolute-complementary-slackness-accuracy         1.0D-5
!  relative-complementary-slackness-accuracy         1.0D-5
!  mininum-initial-primal-feasibility                1000.0
!  mininum-initial-dual-feasibility                  1000.0
!  initial-barrier-parameter                         -1.0
!  feasibility-vs-complementarity-weight             1.0
!  balance-complentarity-factor                      1.0D-5
!  balance-feasibility-factor                        1.0D-5
!  poor-iteration-tolerance                          0.98
!  identical-bounds-tolerance                        1.0D-15
!  required-barrier-value-before-pounce              1.0D-5
!  primal-indicator-tolerance                        1.0D-5
!  primal-dual-indicator-tolerance                   1.0
!  tapia-indicator-tolerance                         0.9
!  maximum-cpu-time-limit                            -1.0
!  maximum-clock-time-limit                          -1.0
!  remove-linear-dependencies                        T
!  treat-zero-bounds-as-general                      F
!  treat-separable-as-general                        F
!  just-find-feasible-point                          F
!  balance-initial-complentarity                     F
!  get-advanced-dual-variables                       F
!  puiseux-series                                    T
!  try-every-order-of-series                         T
!  move-final-solution-onto-bound                    F
!  cross-over-solution                               T
!  solve-reduced-pounce-system                       T
!  array-syntax-worse-than-do-loop                   F
!  space-critical                                    F
!  deallocate-error-fatal                            F
!  generate-sif-file                                 F
!  generate-qplib-file                               F
!  symmetric-linear-equation-solver                  ssids
!  sif-file-name                                     BLLSBPROB.SIF
!  qplib-file-name                                   BLLSBPROB.qplib
!  output-line-prefix                                ""
! END BLLSB SPECIFICATIONS (DEFAULT)

!  Dummy arguments

      TYPE ( BLLSB_control_type ), INTENT( INOUT ) :: control
      INTEGER ( KIND = ip_ ), INTENT( IN ) :: device
      CHARACTER( LEN = * ), OPTIONAL :: alt_specname

!  Programming: Nick Gould and Ph. Toint, January 2002.

!  Local variables

      INTEGER ( KIND = ip_ ), PARAMETER :: error = 1
      INTEGER ( KIND = ip_ ), PARAMETER :: out = error + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: print_level = out + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: start_print = print_level + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: stop_print = start_print + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: maxit = stop_print + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: infeas_max = maxit + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: muzero_fixed = infeas_max + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: restore_problem = muzero_fixed + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: indicator_type = restore_problem + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: arc = indicator_type + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: series_order = arc + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: sif_file_device = series_order + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: qplib_file_device                   &
                                             = sif_file_device + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: infinity = qplib_file_device + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: stop_abs_p = infinity + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: stop_rel_p = stop_abs_p + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: stop_abs_d = stop_rel_p + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: stop_rel_d = stop_abs_d + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: stop_abs_c = stop_rel_d + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: stop_rel_c = stop_abs_c + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: prfeas = stop_rel_c + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: dufeas = prfeas + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: muzero = dufeas + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: tau = muzero + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: gamma_c = tau + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: gamma_f = gamma_c + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: reduce_infeas = gamma_f + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: identical_bounds_tol                &
                                             = reduce_infeas + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: mu_pounce = identical_bounds_tol + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: indicator_tol_p = mu_pounce + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: indicator_tol_pd                    &
                                            = indicator_tol_p + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: indicator_tol_tapia                 &
                                            = indicator_tol_pd + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: cpu_time_limit                      &
                                            = indicator_tol_tapia + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: clock_time_limit = cpu_time_limit + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: remove_dependencies &
                                            = clock_time_limit + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: treat_zero_bounds_as_general        &
                                             = remove_dependencies + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: treat_separable_as_general          &
                                             = treat_zero_bounds_as_general + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: just_feasible                       &
                                             = treat_separable_as_general + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: getdua = just_feasible + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: puiseux = getdua + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: every_order = puiseux + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: feasol = every_order + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: balance_initial_complentarity       &
                                             = feasol + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: crossover                           &
                                             = balance_initial_complentarity + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: space_critical = crossover + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: deallocate_error_fatal              &
                                             = space_critical + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: generate_sif_file                   &
                                             = deallocate_error_fatal + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: generate_qplib_file                 &
                                             = generate_sif_file + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: symmetric_linear_solver             &
                                             = generate_qplib_file + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: sif_file_name                       &
                                             = symmetric_linear_solver + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: qplib_file_name = sif_file_name + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: prefix = qplib_file_name + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: lspec = prefix
      CHARACTER( LEN = 5 ), PARAMETER :: specname = 'BLLSB'
      TYPE ( SPECFILE_item_type ), DIMENSION( lspec ) :: spec

!  Define the keywords

!  Integer key-words

      spec( error )%keyword = 'error-printout-device'
      spec( out )%keyword = 'printout-device'
      spec( print_level )%keyword = 'print-level'
      spec( start_print )%keyword = 'start-print'
      spec( stop_print )%keyword = 'stop-print'
      spec( maxit )%keyword = 'maximum-number-of-iterations'
      spec( infeas_max )%keyword = 'maximum-poor-iterations-before-infeasible'
      spec( muzero_fixed )%keyword = 'barrier-fixed-until-iteration'
      spec( restore_problem )%keyword = 'restore-problem-on-output'
      spec( indicator_type )%keyword = 'indicator-type-used'
      spec( arc )%keyword = 'arc-used'
      spec( series_order )%keyword = 'series-order'
      spec( sif_file_device )%keyword = 'sif-file-device'
      spec( qplib_file_device )%keyword = 'qplib-file-device'

!  Real key-words

      spec( infinity )%keyword = 'infinity-value'
      spec( stop_abs_p )%keyword = 'absolute-primal-accuracy'
      spec( stop_rel_p )%keyword = 'relative-primal-accuracy'
      spec( stop_abs_d )%keyword = 'absolute-dual-accuracy'
      spec( stop_rel_d )%keyword = 'relative-dual-accuracy'
      spec( stop_abs_c )%keyword = 'absolute-complementary-slackness-accuracy'
      spec( stop_rel_c )%keyword = 'relative-complementary-slackness-accuracy'
      spec( prfeas )%keyword = 'mininum-initial-primal-feasibility'
      spec( dufeas )%keyword = 'mininum-initial-dual-feasibility'
      spec( muzero )%keyword = 'initial-barrier-parameter'
      spec( tau )%keyword = 'feasibility-vs-complementarity-weight'
      spec( gamma_c )%keyword = 'balance-complentarity-factor'
      spec( gamma_f )%keyword = 'balance-feasibility-factor'
      spec( reduce_infeas )%keyword = 'poor-iteration-tolerance'
      spec( identical_bounds_tol )%keyword = 'identical-bounds-tolerance'
      spec( mu_pounce )%keyword = 'required-barrier-value-before-pounce'
      spec( indicator_tol_p )%keyword = 'primal-indicator-tolerance'
      spec( indicator_tol_pd )%keyword = 'primal-dual-indicator-tolerance'
      spec( indicator_tol_tapia )%keyword = 'tapia-indicator-tolerance'
      spec( cpu_time_limit )%keyword = 'maximum-cpu-time-limit'
      spec( clock_time_limit )%keyword = 'maximum-clock-time-limit'

!  Logical key-words

      spec( remove_dependencies )%keyword = 'remove-linear-dependencies'
      spec( treat_zero_bounds_as_general )%keyword =                           &
        'treat-zero-bounds-as-general'
      spec( treat_separable_as_general )%keyword = 'treat-separable-as-general'
      spec( just_feasible )%keyword = 'just-find-feasible-point'
      spec( getdua )%keyword = 'get-advanced-dual-variables'
      spec( puiseux )%keyword = 'puiseux-series'
      spec( every_order )%keyword = 'try-every-order-of-series'
      spec( feasol )%keyword = 'move-final-solution-onto-bound'
      spec( balance_initial_complentarity )%keyword =                          &
        'balance-initial-complentarity'
      spec( crossover )%keyword = 'cross-over-solution'
      spec( space_critical )%keyword = 'space-critical'
      spec( deallocate_error_fatal )%keyword = 'deallocate-error-fatal'
      spec( generate_sif_file )%keyword = 'generate-sif-file'
      spec( generate_qplib_file )%keyword = 'generate-qplib-file'

!  Character key-words

      spec( symmetric_linear_solver )%keyword =                                &
        'symmetric-linear-equation-solver'
      spec( sif_file_name )%keyword = 'sif-file-name'
      spec( qplib_file_name )%keyword = 'qplib-file-name'
      spec( prefix )%keyword = 'output-line-prefix'

!     IF ( PRESENT( alt_specname ) ) WRITE(6,*) ' bllsb: ', alt_specname

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
     CALL SPECFILE_assign_value( spec( start_print ),                          &
                                 control%start_print,                          &
                                 control%error )
     CALL SPECFILE_assign_value( spec( stop_print ),                           &
                                 control%stop_print,                           &
                                 control%error )
     CALL SPECFILE_assign_value( spec( maxit ),                                &
                                 control%maxit,                                &
                                 control%error )
     CALL SPECFILE_assign_value( spec( infeas_max ),                           &
                                 control%infeas_max,                           &
                                 control%error )
     CALL SPECFILE_assign_value( spec( muzero_fixed ),                         &
                                 control%muzero_fixed,                         &
                                 control%error )
     CALL SPECFILE_assign_value( spec( restore_problem ),                      &
                                 control%restore_problem,                      &
                                 control%error )
     CALL SPECFILE_assign_value( spec( indicator_type ),                       &
                                 control%indicator_type,                       &
                                 control%error )
     CALL SPECFILE_assign_value( spec( arc ),                                  &
                                 control%arc,                                  &
                                 control%error )
     CALL SPECFILE_assign_value( spec( series_order ),                         &
                                 control%series_order,                         &
                                 control%error )
     CALL SPECFILE_assign_value( spec( sif_file_device ),                      &
                                 control%sif_file_device,                      &
                                 control%error )
     CALL SPECFILE_assign_value( spec( qplib_file_device ),                    &
                                 control%qplib_file_device,                    &
                                 control%error )

!  Set real values

     CALL SPECFILE_assign_value( spec( infinity ),                             &
                                 control%infinity,                             &
                                 control%error )
     CALL SPECFILE_assign_value( spec( stop_abs_p ),                           &
                                 control%stop_abs_p,                           &
                                 control%error )
     CALL SPECFILE_assign_value( spec( stop_rel_p ),                           &
                                 control%stop_rel_p,                           &
                                 control%error )
     CALL SPECFILE_assign_value( spec( stop_abs_d ),                           &
                                 control%stop_abs_d,                           &
                                 control%error )
     CALL SPECFILE_assign_value( spec( stop_rel_d ),                           &
                                 control%stop_rel_d,                           &
                                 control%error )
     CALL SPECFILE_assign_value( spec( stop_abs_c ),                           &
                                 control%stop_abs_c,                           &
                                 control%error )
     CALL SPECFILE_assign_value( spec( stop_rel_c ),                           &
                                 control%stop_rel_c,                           &
                                 control%error )
     CALL SPECFILE_assign_value( spec( prfeas ),                               &
                                 control%prfeas,                               &
                                 control%error )
     CALL SPECFILE_assign_value( spec( dufeas ),                               &
                                 control%dufeas,                               &
                                 control%error )
     CALL SPECFILE_assign_value( spec( muzero ),                               &
                                 control%muzero,                               &
                                 control%error )
     CALL SPECFILE_assign_value( spec( tau ),                                  &
                                 control%tau,                                  &
                                 control%error )
     CALL SPECFILE_assign_value( spec( gamma_c ),                              &
                                 control%gamma_c,                              &
                                 control%error )
     CALL SPECFILE_assign_value( spec( gamma_f ),                              &
                                 control%gamma_f,                              &
                                 control%error )
     CALL SPECFILE_assign_value( spec( reduce_infeas ),                        &
                                 control%reduce_infeas,                        &
                                 control%error )
     CALL SPECFILE_assign_value( spec( identical_bounds_tol ),                 &
                                 control%identical_bounds_tol,                 &
                                 control%error )
     CALL SPECFILE_assign_value( spec( mu_pounce ),                            &
                                 control%mu_pounce,                            &
                                 control%error )
     CALL SPECFILE_assign_value( spec( indicator_tol_p ),                      &
                                 control%indicator_tol_p,                      &
                                 control%error )
     CALL SPECFILE_assign_value( spec( indicator_tol_pd ),                     &
                                 control%indicator_tol_pd,                     &
                                 control%error )
     CALL SPECFILE_assign_value( spec( indicator_tol_tapia ),                  &
                                 control%indicator_tol_tapia,                  &
                                 control%error )
     CALL SPECFILE_assign_value( spec( cpu_time_limit ),                       &
                                 control%cpu_time_limit,                       &
                                 control%error )
     CALL SPECFILE_assign_value( spec( clock_time_limit ),                     &
                                 control%clock_time_limit,                     &
                                 control%error )

!  Set logical values

     CALL SPECFILE_assign_value( spec( remove_dependencies ),                  &
                                 control%remove_dependencies,                  &
                                 control%error )
     CALL SPECFILE_assign_value( spec( treat_zero_bounds_as_general ),         &
                                 control%treat_zero_bounds_as_general,         &
                                 control%error )
     CALL SPECFILE_assign_value( spec( just_feasible ),                        &
                                 control%just_feasible,                        &
                                 control%error )
     CALL SPECFILE_assign_value( spec( treat_separable_as_general ),           &
                                 control%treat_separable_as_general,           &
                                 control%error )
     CALL SPECFILE_assign_value( spec( getdua ),                               &
                                 control%getdua,                               &
                                 control%error )
     CALL SPECFILE_assign_value( spec( puiseux ),                              &
                                 control%puiseux,                              &
                                 control%error )
     CALL SPECFILE_assign_value( spec( every_order ),                          &
                                 control%every_order,                          &
                                 control%error )
     CALL SPECFILE_assign_value( spec( feasol ),                               &
                                 control%feasol,                               &
                                 control%error )
     CALL SPECFILE_assign_value( spec( balance_initial_complentarity ),        &
                                 control%balance_initial_complentarity,        &
                                 control%error )
     CALL SPECFILE_assign_value( spec( crossover ),                            &
                                 control%crossover,                            &
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
     CALL SPECFILE_assign_value( spec( generate_qplib_file ),                  &
                                 control%generate_qplib_file,                  &
                                 control%error )

!  Set character values

     CALL SPECFILE_assign_value( spec( symmetric_linear_solver ),              &
                                 control%symmetric_linear_solver,              &
                                 control%error )
     CALL SPECFILE_assign_value( spec( sif_file_name ),                        &
                                 control%sif_file_name,                        &
                                 control%error )
     CALL SPECFILE_assign_value( spec( qplib_file_name ),                      &
                                 control%qplib_file_name,                      &
                                 control%error )
     CALL SPECFILE_assign_value( spec( prefix ),                               &
                                 control%prefix,                               &
                                 control%error )

!  Read the specfile for FDC

      IF ( PRESENT( alt_specname ) ) THEN
        CALL FDC_read_specfile( control%FDC_control, device,                   &
                                alt_specname = TRIM( alt_specname ) // '-FDC' )
      ELSE
        CALL FDC_read_specfile( control%FDC_control, device )
      END IF
      control%FDC_control%max_infeas = control%stop_abs_p

!  Read the specfiles for SLS and SLS-POUNCE

      IF ( PRESENT( alt_specname ) ) THEN
        CALL SLS_read_specfile( control%SLS_control, device,                   &
                       alt_specname = TRIM( alt_specname ) // '-SLS')
        CALL SLS_read_specfile( control%SLS_pounce_control, device,            &
                       alt_specname = TRIM( alt_specname ) // '-SLS-POUNCE' )
      ELSE
        CALL SLS_read_specfile( control%SLS_control, device )
        CALL SLS_read_specfile( control%SLS_pounce_control, device,            &
                       alt_specname = 'SLS-POUNCE' )
      END IF

!  Read the specfile for FIT

      IF ( PRESENT( alt_specname ) ) THEN
        CALL FIT_read_specfile( control%FIT_control, device,                   &
                                alt_specname = TRIM( alt_specname ) // '-FIT' )
      ELSE
        CALL FIT_read_specfile( control%FIT_control, device )
      END IF

!  Read the specfile for CRO

      IF ( PRESENT( alt_specname ) ) THEN
        CALL CRO_read_specfile( control%CRO_control, device,                   &
                                alt_specname = TRIM( alt_specname ) // '-CRO' )
      ELSE
        CALL CRO_read_specfile( control%CRO_control, device )
      END IF

!  Read the specfile for ROOTS

      IF ( PRESENT( alt_specname ) ) THEN
        CALL ROOTS_read_specfile( control%ROOTS_control, device,               &
                              alt_specname = TRIM( alt_specname ) // '-ROOTS' )
      ELSE
        CALL ROOTS_read_specfile( control%ROOTS_control, device )
      END IF

      RETURN

      END SUBROUTINE BLLSB_read_specfile

!-*-*-*-*-*-*-*-*-   B L L S B _ S O L V E  S U B R O U T I N E   -*-*-*-*-*-*-

      SUBROUTINE BLLSB_solve( prob, data, control, inform,                     &
                              regularization_weight, W )

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
!
!  Minimize the linear least-squares objective function
!
!         1/2 || A_o x - b ||_W^2 + 1/2 weight || x ||^2
!
!  where
!
!            (x_l)_i <=  x_i <= (x_u)_i , i = 1, .... , n,
!
!  x is a vector of n components ( x_1, .... , x_n ),  A_o is an o by n
!  matrix, any of the bounds (x_l)_i, (x_u)_i may be infinite, and the
!  weighted norm ||v||_W = sqrt( sum_i=1^o w_i v_i^2 ), using a primal-dual
!  interior-point method. The subroutine is particularly appropriate
!  when A_0 is sparse.
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
!    to be solved since the last call to BLLSB_initialize, and .FALSE. if
!    a previous call to a problem with the same "structure" (but different
!    numerical data) was made.
!
!   %n is an INTEGER variable, which must be set by the user to the
!    number of optimization parameters, n.  RESTRICTION: %n >= 1
!
!   %o is an INTEGER variable, which must be set by the user to the
!    number of observations, o.  RESTRICTION: o >= 1
!
!   %Ao is a structure of type SMT_type used to hold the design matrix A_o.
!    Five storage formats are permitted:
!
!    i) sparse, co-ordinate
!
!       In this case, the following must be set:
!
!       %Ao%type( 1 : 10 ) = TRANSFER( 'COORDINATE', %Ao%type )
!       %Ao%val( : ) the values of the components of A_o
!       %Ao%row( : ) the row indices of the components of A_o
!       %Ao%col( : ) the column indices of the components of A_o
!       %Ao%ne       the number of nonzeros used to store A_o
!
!    ii) sparse, by rows
!
!       In this case, the following must be set:
!
!       %Ao%type( 1 : 14 ) = TRANSFER( 'SPARSE_BY_ROWS', %Ao%type )
!       %Ao%val( : ) the values of the components of A_o, stored row by row
!       %Ao%col( : ) the column indices of the components of A_o
!       %Ao%ptr( : ) pointers to the start of each row, and past the end of
!                    the last row
!
!    iii) sparse, by columns
!
!       In this case, the following must be set:
!
!       %Ao%type( 1 : 17 ) = TRANSFER( 'SPARSE_BY_COLUMNS', %Ao%type )
!       %Ao%val( : ) the values of the components of A_o, stored column
!                    by column
!       %Ao%row( : ) the row indices of the components of A_o
!       %Ao%ptr( : ) pointers to the start of each column, and past the end of
!                    the last column
!
!    iv) dense, by rows
!
!       In this case, the following must be set:
!
!       %Ao%type( 1 : 5 ) = TRANSFER( 'DENSE', %Ao%type )
!         [ or %Ao%type( 1 : 13 ) = TRANSFER( 'DENSE_BY_ROWS', %Ao%type ) ]
!       %Ao%val( : ) the values of the components of A, stored row by row,
!                    with each the entries in each row in order of
!                    increasing column indicies.
!
!    v) dense, by columns
!
!       In this case, the following must be set:
!
!       %Ao%type( 1 : 16 ) = TRANSFER( 'DENSE_BY_COLUMNS', %Ao%type )
!       %Ao%val( : ) the values of the components of A_o, stored column
!                    by column with each the entries in each column in order
!                    of increasing row indicies.
!
!   %B is a REAL array of length o, which must be set by the user to the value
!    of the observations, b. The i-th component of B, i = 1, ...., o should
!    contain the value of b_i.
!
!   %X is a REAL array of length %n, which must be set by the user
!    to estimaes of the solution, x. On successful exit, it will contain
!    the required solution, x.
!
!   %R is a REAL array of length %o, which is used to store the values of
!    the residuals A_o x - b. It need not be set on entry. On exit, it will
!    have been filled with appropriate values.
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
!   %X_status is an INTEGER array of length %n, which will be set on exit to
!    indicate the likely ultimate status of the simple bound constraints.
!    Possible values are
!    X_status( i ) < 0, the i-th bound constraint is likely in the active set,
!                       on its lower bound,
!                  > 0, the i-th bound constraint is likely in the active set
!                       on its upper bound, and
!                  = 0, the i-th bound constraint is likely not in the active
!                       set
!    It need not be set on entry.
!
!  data is a structure of type BLLSB_data_type which holds private internal data
!
!  control is a structure of type BLLSB_control_type that controls the
!   execution of the subroutine and must be set by the user. Default values for
!   the elements may be set by a call to BLLSB_initialize. See the preamble
!   for details
!
!  inform is a structure of type BLLSB_inform_type that provides
!    information on exit from BLLSB_solve. The component status
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
!        prob%o     >=  1
!        prob%Ao%type in { 'DENSE', 'DENSE_BY_COLUMNS', 'SPARSE_BY_ROWS',
!                          'SPARSE_BY_COLUMNS','COORDINATE' }
!       has been violated.
!
!    -5 The constraints have no feasible point.
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
!   -18 Too many iterations have been performed. This may happen if
!       control%maxit is too small, but may also be symptomatic of
!       a badly scaled problem.
!
!   -19 Too much time has passed. This may happen if control%cpu_time_limit or
!       control%clock_time_limit is too small, but may also be symptomatic of
!       a badly scaled problem.
!
!  On exit from BLLSB_solve, other components of inform are given in the
!   preamble to CLLS
!
!  regularization_weight is an OPTIONAL REAL, that may be set by the user
!   to the value of the non-negative regularization weight. If it is absent,
!   the regularization weight will be zero.
!
!  W is an OPTIONAL REAL array of length prob%o, that may be set by the user
!   to the values of the components of the weights W. If it is absent,
!   the weights will all be taken to be 1.0.
!
! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!  Dummy arguments

      TYPE ( QPT_problem_type ), INTENT( INOUT ) :: prob
      TYPE ( BLLSB_data_type ), INTENT( INOUT ) :: data
      TYPE ( BLLSB_control_type ), INTENT( IN ) :: control
      TYPE ( BLLSB_inform_type ), INTENT( OUT ) :: inform

!  optional dummy argument

      REAL ( KIND = rp_ ), OPTIONAL, INTENT( IN ) :: regularization_weight
      REAL ( KIND = rp_ ), OPTIONAL, INTENT( IN ), DIMENSION( prob%o ) :: W

!  Local variables

      INTEGER ( KIND = ip_ ) :: status, alloc_status
      REAL :: time_start, time_now
      REAL ( KIND = rp_ ) :: clock_start, clock_now
      CHARACTER ( LEN = 80 ) :: array_name, bad_alloc

!  initialize time

      CALL CPU_TIME( time_start ) ; CALL CLOCK_time( clock_start )

!  set up a null constraint matrix A

      prob%m = 0 ; prob%A%ne = 0 ; prob%A%m = 0 ; prob%A%n = prob%n

      CALL SMT_put( prob%A%type, 'COORDINATE', inform%alloc_status )
      IF ( inform%alloc_status /= 0 ) THEN
        inform%status = GALAHAD_error_allocate ; GO TO 910
      END IF

!  allocate space for the null A and the ancillary c and y

      array_name = 'bllsb: prob%A%row'
      CALL SPACE_resize_array( prob%A%ne, prob%A%row, inform%status,           &
             inform%alloc_status, array_name = array_name,                     &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 910

      array_name = 'bllsb: prob%A%col'
      CALL SPACE_resize_array( prob%A%ne, prob%A%col, inform%status,           &
             inform%alloc_status, array_name = array_name,                     &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 910

      array_name = 'bllsb: prob%A%val'
      CALL SPACE_resize_array( prob%A%ne, prob%A%val, inform%status,           &
             inform%alloc_status, array_name = array_name,                     &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 910

      array_name = 'bllsb: prob%C'
      CALL SPACE_resize_array( prob%m, prob%C, inform%status,                  &
             inform%alloc_status, array_name = array_name,                     &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 910

      array_name = 'bllsb: prob%C_l'
      CALL SPACE_resize_array( prob%m, prob%C_l, inform%status,                &
             inform%alloc_status, array_name = array_name,                     &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 910

      array_name = 'bllsb: prob%C_u'
      CALL SPACE_resize_array( prob%m, prob%C_u, inform%status,                &
             inform%alloc_status, array_name = array_name,                     &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 910

      array_name = 'bllsb: prob%Y'
      CALL SPACE_resize_array( prob%m, prob%Y, inform%status,                  &
             inform%alloc_status, array_name = array_name,                     &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 910

!  now call the generic interior-point constrained linear least-squares solver

      CALL CLLS_solve( prob, data, control, inform,                            &
                       regularization_weight = regularization_weight, W = W )

!  save status values while arrays are deallocated

      status = inform%status
      alloc_status = inform%alloc_status
      bad_alloc = inform%bad_alloc

!  deallocate space used for null A and its associates

      array_name = 'bllsb: prob%A%row'
      CALL SPACE_dealloc_array( prob%A%row,                                    &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'bllsb: prob%A%col'
      CALL SPACE_dealloc_array( prob%A%col,                                    &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'bllsb: prob%A%ptr'
      CALL SPACE_dealloc_array( prob%A%ptr,                                    &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'bllsb: prob%A%val'
      CALL SPACE_dealloc_array( prob%A%val,                                    &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'bllsb: prob%A%type'
      CALL SPACE_dealloc_array( prob%A%type,                                   &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'bllsb: prob%C'
      CALL SPACE_dealloc_array( prob%C,                                        &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'bllsb: prob%C_l'
      CALL SPACE_dealloc_array( prob%C_l,                                      &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'bllsb: prob%C_u'
      CALL SPACE_dealloc_array( prob%C_u,                                      &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'bllsb: prob%Y'
      CALL SPACE_dealloc_array( prob%Y,                                        &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

!  restore status values

      inform%status = status
      inform%alloc_status = alloc_status
      inform%bad_alloc = bad_alloc

      RETURN

!  error returns

  910 CONTINUE
      CALL CPU_TIME( time_now ) ; CALL CLOCK_time( clock_now )
      inform%time%total = REAL( time_now - time_start, rp_ )
      inform%time%clock_total = clock_now - clock_start
      RETURN

!  end of SUBROUTINE BLLSB_solve

      END SUBROUTINE BLLSB_solve

! -*-*-*-*-*-   B L L S B _ T E R M I N A T E   S U B R O U T I N E   -*-*-*-*-

      SUBROUTINE BLLSB_terminate( data, control, inform )

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!      ..............................................
!      .                                            .
!      .  Deallocate internal arrays at the end     .
!      .  of the computation                        .
!      .                                            .
!      ..............................................

!  Arguments:
!
!   data    see Subroutine BLLSB_initialize
!   control see Subroutine BLLSB_initialize
!   inform  see Subroutine BLLSB_solve

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!  Dummy arguments

      TYPE ( BLLSB_data_type ), INTENT( INOUT ) :: data
      TYPE ( BLLSB_control_type ), INTENT( IN ) :: control
      TYPE ( BLLSB_inform_type ), INTENT( INOUT ) :: inform

      CALL CLLS_terminate( data, control, inform )
      RETURN

!  End of subroutine BLLSB_terminate

      END SUBROUTINE BLLSB_terminate

!  G A L A H A D -  B L L S B _ f u l l _ t e r m i n a t e  S U B R O U T I N E

     SUBROUTINE BLLSB_full_terminate( data, control, inform )

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!   Deallocate all private storage

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     TYPE ( BLLSB_full_data_type ), INTENT( INOUT ) :: data
     TYPE ( BLLSB_control_type ), INTENT( IN ) :: control
     TYPE ( BLLSB_inform_type ), INTENT( INOUT ) :: inform

!-----------------------------------------------
!   L o c a l   V a r i a b l e s
!-----------------------------------------------

     CHARACTER ( LEN = 80 ) :: array_name

!  deallocate workspace

     CALL BLLSB_terminate( data%bllsb_data, control, inform )

!  deallocate any internal problem arrays

     array_name = 'bllsb: data%prob%X'
     CALL SPACE_dealloc_array( data%prob%X,                                    &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'bllsb: data%prob%X_l'
     CALL SPACE_dealloc_array( data%prob%X_l,                                  &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'bllsb: data%prob%X_u'
     CALL SPACE_dealloc_array( data%prob%X_u,                                  &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'bllsb: data%prob%Z'
     CALL SPACE_dealloc_array( data%prob%Z,                                    &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'bllsb: data%prob%R'
     CALL SPACE_dealloc_array( data%prob%R,                                    &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'bllsb: data%prob%C_l'
     CALL SPACE_dealloc_array( data%prob%C_l,                                  &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'bllsb: data%prob%C_u'
     CALL SPACE_dealloc_array( data%prob%C_u,                                  &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'bllsb: data%prob%Ao%ptr'
     CALL SPACE_dealloc_array( data%prob%Ao%ptr,                               &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'bllsb: data%prob%Ao%row'
     CALL SPACE_dealloc_array( data%prob%Ao%row,                               &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'bllsb: data%prob%Ao%col'
     CALL SPACE_dealloc_array( data%prob%Ao%col,                               &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'bllsb: data%prob%Ao%val'
     CALL SPACE_dealloc_array( data%prob%Ao%val,                               &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'bllsb: data%prob%Ao%type'
     CALL SPACE_dealloc_array( data%prob%Ao%type,                              &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'bllsb: data%prob%X_status'
     CALL SPACE_dealloc_array( data%prob%X_status,                             &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     RETURN

!  End of subroutine BLLSB_full_terminate

     END SUBROUTINE BLLSB_full_terminate

! -----------------------------------------------------------------------------
! =============================================================================
! -----------------------------------------------------------------------------
!              specific interfaces to make calls from C easier
! -----------------------------------------------------------------------------
! =============================================================================
! -----------------------------------------------------------------------------

!-*-*-*-  G A L A H A D -  B L L S B _ i m p o r t _ S U B R O U T I N E -*-*-*-

     SUBROUTINE BLLSB_import( control, data, status, n, o,                     &
                              Ao_type, Ao_ne, Ao_row, Ao_col, Ao_ptr )

!  import fixed problem data into internal storage prior to solution.
!  Arguments are as follows:

!  control is a derived type whose components are described in the leading
!   comments to BLLSB_solve
!
!  data is a scalar variable of type BLLSB_full_data_type used for internal data
!
!  status is a scalar variable of type default intege that indicates the
!   success or otherwise of the import. Possible values are:
!
!    1. The import was succesful, and the package is ready for the solve phase
!
!   -1. An allocation error occurred. A message indicating the offending
!       array is written on unit control.error, and the returned allocation
!       status and a string containing the name of the offending array
!       are held in inform.alloc_status and inform.bad_alloc respectively.
!   -2. A deallocation error occurred.  A message indicating the offending
!       array is written on unit control.error and the returned allocation
!       status and a string containing the name of the offending array
!       are held in inform.alloc_status and inform.bad_alloc respectively.
!   -3. The restriction n > 0, o >= 0 or requirement that type contains
!       its relevant string 'DENSE', 'DENSE_BY_ROWS', 'DENSE_BY_COLUMNS',
!       'COORDINATE', 'SPARSE_BY_ROWS' or 'SPARSE_BY_COLUMNS',
!       has been violated.
!
!  n is a scalar variable of type default integer, that holds the number of
!   variables
!
!  o is a scalar variable of type default integer, that holds the number of
!   observations
!
!  Ao_type is a character string that specifies the objective design matrix
!   storage scheme used. It should be one of 'coordinate', 'sparse_by_rows',
!   'sparse_by_columns', 'dense', 'dense_by_rows' or 'dense_by_columns';
!   lower or upper case variants are allowed
!
!  Ao_ne is a scalar variable of type default integer, that holds the number of
!   entries in J in the sparse co-ordinate storage scheme. It need not be set
!  for any of the other schemes.
!
!  Ao_row is a rank-one array of type default integer, that holds the row
!   indices J in the sparse co-ordinate storage scheme. It need not be set
!   for any of the other schemes, and in this case can be of length 0
!
!  Ao_col is a rank-one array of type default integer, that holds the column
!   indices of J in either the sparse co-ordinate, or the sparse row-wise
!   storage scheme. It need not be set when the dense schemes are used, and
!   in this case can be of length 0
!
!  Ao_ptr is a rank-one array of dimension n+1 and type default integer,
!   that holds the starting position of each row of J, as well as the total
!   number of entries plus one, in the sparse row-wise storage scheme.
!   It need not be set when the other schemes are used, and in this case
!   can be of length 0
!
!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     TYPE ( BLLSB_control_type ), INTENT( INOUT ) :: control
     TYPE ( BLLSB_full_data_type ), INTENT( INOUT ) :: data
     INTEGER ( KIND = ip_ ), INTENT( IN ) :: n, o, Ao_ne
     INTEGER ( KIND = ip_ ), INTENT( OUT ) :: status
     CHARACTER ( LEN = * ), INTENT( IN ) :: Ao_type
     INTEGER ( KIND = ip_ ), DIMENSION( : ), OPTIONAL, INTENT( IN ) :: Ao_row
     INTEGER ( KIND = ip_ ), DIMENSION( : ), OPTIONAL, INTENT( IN ) :: Ao_col
     INTEGER ( KIND = ip_ ), DIMENSION( : ), OPTIONAL, INTENT( IN ) :: Ao_ptr

!  local variables

     INTEGER ( KIND = ip_ ) :: error
     LOGICAL :: deallocate_error_fatal, space_critical
     CHARACTER ( LEN = 80 ) :: array_name

!  copy control to data

     WRITE( control%out, "( '' )", ADVANCE = 'no') ! prevents ifort bug
     data%bllsb_control = control

     error = data%bllsb_control%error
     space_critical = data%bllsb_control%space_critical
     deallocate_error_fatal = data%bllsb_control%space_critical

!  allocate vector space if required

     array_name = 'bllsb: data%prob%B'
     CALL SPACE_resize_array( o, data%prob%B,                                  &
            data%bllsb_inform%status, data%bllsb_inform%alloc_status,          &
            array_name = array_name,                                           &
            deallocate_error_fatal =                                           &
              data%bllsb_control%deallocate_error_fatal,                       &
            exact_size = data%bllsb_control%space_critical,                    &
            bad_alloc = data%bllsb_inform%bad_alloc,                           &
            out = data%bllsb_control%error )
     IF ( data%bllsb_inform%status /= 0 ) GO TO 900

     array_name = 'bllsb: data%prob%X_l'
     CALL SPACE_resize_array( n, data%prob%X_l,                                &
            data%bllsb_inform%status, data%bllsb_inform%alloc_status,          &
            array_name = array_name,                                           &
            deallocate_error_fatal = deallocate_error_fatal,                   &
            exact_size = space_critical,                                       &
            bad_alloc = data%bllsb_inform%bad_alloc, out = error )
     IF ( data%bllsb_inform%status /= 0 ) GO TO 900

     array_name = 'bllsb: data%prob%X_u'
     CALL SPACE_resize_array( n, data%prob%X_u,                                &
            data%bllsb_inform%status, data%bllsb_inform%alloc_status,          &
            array_name = array_name,                                           &
            deallocate_error_fatal = deallocate_error_fatal,                   &
            exact_size = space_critical,                                       &
            bad_alloc = data%bllsb_inform%bad_alloc, out = error )
     IF ( data%bllsb_inform%status /= 0 ) GO TO 900

     array_name = 'bllsb: data%prob%X'
     CALL SPACE_resize_array( n, data%prob%X,                                  &
            data%bllsb_inform%status, data%bllsb_inform%alloc_status,          &
            array_name = array_name,                                           &
            deallocate_error_fatal = deallocate_error_fatal,                   &
            exact_size = space_critical,                                       &
            bad_alloc = data%bllsb_inform%bad_alloc, out = error )
     IF ( data%bllsb_inform%status /= 0 ) GO TO 900

     array_name = 'bllsb: data%prob%R'
     CALL SPACE_resize_array( n, data%prob%R,                                  &
            data%bllsb_inform%status, data%bllsb_inform%alloc_status,          &
            array_name = array_name,                                           &
            deallocate_error_fatal = deallocate_error_fatal,                   &
            exact_size = space_critical,                                       &
            bad_alloc = data%bllsb_inform%bad_alloc, out = error )
     IF ( data%bllsb_inform%status /= 0 ) GO TO 900

     array_name = 'bllsb: data%prob%Z'
     CALL SPACE_resize_array( n, data%prob%Z,                                  &
            data%bllsb_inform%status, data%bllsb_inform%alloc_status,          &
            array_name = array_name,                                           &
            deallocate_error_fatal = deallocate_error_fatal,                   &
            exact_size = space_critical,                                       &
            bad_alloc = data%bllsb_inform%bad_alloc, out = error )
     IF ( data%bllsb_inform%status /= 0 ) GO TO 900

     array_name = 'bllsb: data%prob%X_status'
     CALL SPACE_resize_array( n, data%prob%X_status,                           &
            data%bllsb_inform%status, data%bllsb_inform%alloc_status,          &
            array_name = array_name,                                           &
            deallocate_error_fatal = deallocate_error_fatal,                   &
            exact_size = space_critical,                                       &
            bad_alloc = data%bllsb_inform%bad_alloc, out = error )
     IF ( data%bllsb_inform%status /= 0 ) GO TO 900

!  put data into the required components of the qpt storage type

     data%prob%n = n ; data%prob%o = o

!  set Ao appropriately in the qpt storage type

     SELECT CASE ( Ao_type )
     CASE ( 'coordinate', 'COORDINATE' )
       IF ( .NOT. ( PRESENT( Ao_row ) .AND. PRESENT( Ao_col ) ) ) THEN
         data%bllsb_inform%status = GALAHAD_error_optional
         GO TO 900
       END IF
       CALL SMT_put( data%prob%Ao%type, 'COORDINATE',                          &
                     data%bllsb_inform%alloc_status )
       data%prob%Ao%n = n ; data%prob%Ao%m = o
       data%prob%Ao%ne = Ao_ne

       array_name = 'bllsb: data%prob%Ao%row'
       CALL SPACE_resize_array( data%prob%Ao%ne, data%prob%Ao%row,             &
              data%bllsb_inform%status, data%bllsb_inform%alloc_status,        &
              array_name = array_name,                                         &
              deallocate_error_fatal = deallocate_error_fatal,                 &
              exact_size = space_critical,                                     &
              bad_alloc = data%bllsb_inform%bad_alloc, out = error )
       IF ( data%bllsb_inform%status /= 0 ) GO TO 900

       array_name = 'bllsb: data%prob%Ao%col'
       CALL SPACE_resize_array( data%prob%Ao%ne, data%prob%Ao%col,             &
              data%bllsb_inform%status, data%bllsb_inform%alloc_status,        &
              array_name = array_name,                                         &
              deallocate_error_fatal = deallocate_error_fatal,                 &
              exact_size = space_critical,                                     &
              bad_alloc = data%bllsb_inform%bad_alloc, out = error )
       IF ( data%bllsb_inform%status /= 0 ) GO TO 900

       array_name = 'bllsb: data%prob%Ao%val'
       CALL SPACE_resize_array( data%prob%Ao%ne, data%prob%Ao%val,             &
              data%bllsb_inform%status, data%bllsb_inform%alloc_status,        &
              array_name = array_name,                                         &
              deallocate_error_fatal = deallocate_error_fatal,                 &
              exact_size = space_critical,                                     &
              bad_alloc = data%bllsb_inform%bad_alloc, out = error )
       IF ( data%bllsb_inform%status /= 0 ) GO TO 900
       IF ( data%f_indexing ) THEN
         data%prob%Ao%row( : data%prob%Ao%ne ) = Ao_row( : data%prob%Ao%ne )
         data%prob%Ao%col( : data%prob%Ao%ne ) = Ao_col( : data%prob%Ao%ne )
       ELSE
         data%prob%Ao%row( : data%prob%Ao%ne ) = Ao_row( : data%prob%Ao%ne ) + 1
         data%prob%Ao%col( : data%prob%Ao%ne ) = Ao_col( : data%prob%Ao%ne ) + 1
       END IF

     CASE ( 'sparse_by_rows', 'SPARSE_BY_ROWS' )
       IF ( .NOT. ( PRESENT( Ao_ptr ) .AND. PRESENT( Ao_col ) ) ) THEN
         data%bllsb_inform%status = GALAHAD_error_optional
         GO TO 900
       END IF
       CALL SMT_put( data%prob%Ao%type, 'SPARSE_BY_ROWS',                      &
                     data%bllsb_inform%alloc_status )
       data%prob%Ao%n = n ; data%prob%Ao%m = o
       IF ( data%f_indexing ) THEN
         data%prob%Ao%ne = Ao_ptr( o + 1 ) - 1
       ELSE
         data%prob%Ao%ne = Ao_ptr( o + 1 )
       END IF
       array_name = 'bllsb: data%prob%Ao%ptr'
       CALL SPACE_resize_array( o + 1, data%prob%Ao%ptr,                       &
              data%bllsb_inform%status, data%bllsb_inform%alloc_status,        &
              array_name = array_name,                                         &
              deallocate_error_fatal = deallocate_error_fatal,                 &
              exact_size = space_critical,                                     &
              bad_alloc = data%bllsb_inform%bad_alloc, out = error )
       IF ( data%bllsb_inform%status /= 0 ) GO TO 900

       array_name = 'bllsb: data%prob%Ao%col'
       CALL SPACE_resize_array( data%prob%Ao%ne, data%prob%Ao%col,             &
              data%bllsb_inform%status, data%bllsb_inform%alloc_status,        &
              array_name = array_name,                                         &
              deallocate_error_fatal = deallocate_error_fatal,                 &
              exact_size = space_critical,                                     &
              bad_alloc = data%bllsb_inform%bad_alloc, out = error )
       IF ( data%bllsb_inform%status /= 0 ) GO TO 900

       array_name = 'bllsb: data%prob%Ao%val'
       CALL SPACE_resize_array( data%prob%Ao%ne, data%prob%Ao%val,             &
              data%bllsb_inform%status, data%bllsb_inform%alloc_status,        &
              array_name = array_name,                                         &
              deallocate_error_fatal = deallocate_error_fatal,                 &
              exact_size = space_critical,                                     &
              bad_alloc = data%bllsb_inform%bad_alloc, out = error )
       IF ( data%bllsb_inform%status /= 0 ) GO TO 900

       IF ( data%f_indexing ) THEN
         data%prob%Ao%ptr( : o + 1 ) = Ao_ptr( : o + 1 )
         data%prob%Ao%col( : data%prob%Ao%ne ) = Ao_col( : data%prob%Ao%ne )
       ELSE
         data%prob%Ao%ptr( : o + 1 ) = Ao_ptr( : o + 1 ) + 1
         data%prob%Ao%col( : data%prob%Ao%ne ) = Ao_col( : data%prob%Ao%ne ) + 1
       END IF

     CASE ( 'sparse_by_columns', 'SPARSE_BY_COLUMNS' )
       IF ( .NOT. ( PRESENT( Ao_ptr ) .AND. PRESENT( Ao_row ) ) ) THEN
         data%bllsb_inform%status = GALAHAD_error_optional
         GO TO 900
       END IF
       CALL SMT_put( data%prob%Ao%type, 'SPARSE_BY_COLUMNS',                   &
                     data%bllsb_inform%alloc_status )
       data%prob%Ao%n = n ; data%prob%Ao%m = o
       IF ( data%f_indexing ) THEN
         data%prob%Ao%ne = Ao_ptr( n + 1 ) - 1
       ELSE
         data%prob%Ao%ne = Ao_ptr( n + 1 )
       END IF
       array_name = 'bllsb: data%prob%Ao%ptr'
       CALL SPACE_resize_array( n + 1, data%prob%Ao%ptr,                       &
              data%bllsb_inform%status, data%bllsb_inform%alloc_status,        &
              array_name = array_name,                                         &
              deallocate_error_fatal = deallocate_error_fatal,                 &
              exact_size = space_critical,                                     &
              bad_alloc = data%bllsb_inform%bad_alloc, out = error )
       IF ( data%bllsb_inform%status /= 0 ) GO TO 900

       array_name = 'bllsb: data%prob%Ao%row'
       CALL SPACE_resize_array( data%prob%Ao%ne, data%prob%Ao%row,             &
              data%bllsb_inform%status, data%bllsb_inform%alloc_status,        &
              array_name = array_name,                                         &
              deallocate_error_fatal = deallocate_error_fatal,                 &
              exact_size = space_critical,                                     &
              bad_alloc = data%bllsb_inform%bad_alloc, out = error )
       IF ( data%bllsb_inform%status /= 0 ) GO TO 900

       array_name = 'bllsb: data%prob%Ao%val'
       CALL SPACE_resize_array( data%prob%Ao%ne, data%prob%Ao%val,             &
              data%bllsb_inform%status, data%bllsb_inform%alloc_status,        &
              array_name = array_name,                                         &
              deallocate_error_fatal = deallocate_error_fatal,                 &
              exact_size = space_critical,                                     &
              bad_alloc = data%bllsb_inform%bad_alloc, out = error )
       IF ( data%bllsb_inform%status /= 0 ) GO TO 900

       IF ( data%f_indexing ) THEN
         data%prob%Ao%ptr( : n + 1 ) = Ao_ptr( : n + 1 )
         data%prob%Ao%row( : data%prob%Ao%ne ) = Ao_row( : data%prob%Ao%ne )
       ELSE
         data%prob%Ao%ptr( : n + 1 ) = Ao_ptr( : n + 1 ) + 1
         data%prob%Ao%row( : data%prob%Ao%ne ) = Ao_row( : data%prob%Ao%ne ) + 1
       END IF

     CASE ( 'dense', 'DENSE', 'dense_by_rows', 'DENSE_BY_ROWS' )
       CALL SMT_put( data%prob%Ao%type, 'DENSE',                               &
                     data%bllsb_inform%alloc_status )
       data%prob%Ao%n = n ; data%prob%Ao%m = o
       data%prob%Ao%ne = o * n

       array_name = 'bllsb: data%prob%Ao%val'
       CALL SPACE_resize_array( data%prob%Ao%ne, data%prob%Ao%val,             &
              data%bllsb_inform%status, data%bllsb_inform%alloc_status,        &
              array_name = array_name,                                         &
              deallocate_error_fatal = deallocate_error_fatal,                 &
              exact_size = space_critical,                                     &
              bad_alloc = data%bllsb_inform%bad_alloc, out = error )
       IF ( data%bllsb_inform%status /= 0 ) GO TO 900

     CASE ( 'dense_by_columns', 'DENSE_BY_COLUMNS' )
       CALL SMT_put( data%prob%Ao%type, 'DENSE_BY_COLUMNS',                    &
                     data%bllsb_inform%alloc_status )
       data%prob%Ao%n = n ; data%prob%Ao%m = o
       data%prob%Ao%ne = o * n

       array_name = 'bllsb: data%prob%Ao%val'
       CALL SPACE_resize_array( data%prob%Ao%ne, data%prob%Ao%val,             &
              data%bllsb_inform%status, data%bllsb_inform%alloc_status,        &
              array_name = array_name,                                         &
              deallocate_error_fatal = deallocate_error_fatal,                 &
              exact_size = space_critical,                                     &
              bad_alloc = data%bllsb_inform%bad_alloc, out = error )
       IF ( data%bllsb_inform%status /= 0 ) GO TO 900

     CASE DEFAULT
       data%bllsb_inform%status = GALAHAD_error_unknown_storage
       GO TO 900
     END SELECT

     status = GALAHAD_ok
     RETURN

!  error returns

 900 CONTINUE
     status = data%bllsb_inform%status
     RETURN

!  End of subroutine BLLSB_import

     END SUBROUTINE BLLSB_import

!-  G A L A H A D -  B L L S B _ r e s e t _ c o n t r o l   S U B R O U T I N E

     SUBROUTINE BLLSB_reset_control( control, data, status )

!  reset control parameters after import if required.
!  See BLLSB_solve for a description of the required arguments

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     TYPE ( BLLSB_control_type ), INTENT( IN ) :: control
     TYPE ( BLLSB_full_data_type ), INTENT( INOUT ) :: data
     INTEGER ( KIND = ip_ ), INTENT( OUT ) :: status

!  set control in internal data

     data%bllsb_control = control

!  flag a successful call

     status = GALAHAD_ok
     RETURN

!  end of subroutine BLLSB_reset_control

     END SUBROUTINE BLLSB_reset_control

!-*-  G A L A H A D -  B L L S B _ s o l v e _ b l l s   S U B R O U T I N E -*-

     SUBROUTINE BLLSB_solve_blls( data, status, Ao_val, B, X_l, X_u, X, R, Z,  &
                                  X_stat, regularization_weight, W )

!  solve the constrained linear least-squares problem whose structure was
!  previously imported. See BLLSB_solve for a description of the required
!  arguments.

!--------------------------------
!   D u m m y   A r g u m e n t s
!--------------------------------

!  data is a scalar variable of type BLLSB_full_data_type used for internal data
!
!  status is a scalar variable of type default intege that indicates the
!   success or otherwise of the import. If status = 0, the solve was succesful.
!   For other values see, bllsb_solve above.
!
!  Ao_val is a rank-one array of type default real, that holds the values
!   of the design matrix Ao in the storage scheme specified in bllsb_import.
!
!  B is a rank-one array of dimension o and type default
!   real, that holds the vector of linear terms of the observations, b.
!   The i-th component of B, i = 1, ... , o, contains (b)_i.
!
!  X_l, X_u are rank-one arrays of dimension n, that hold the values of
!   the lower and upper bounds, c_l and c_u, on the variables x.
!   Any bound x_l(i) or x_u(i) larger than or equal to control%infinity in
!   absolute value will be regarded as being infinite (see the entry
!   control%infinity). Thus, an infinite lower bound may be specified by
!   setting the appropriate component of X_l to a value smaller than
!   -control%infinity, while an infinite upper bound can be specified by
!   setting the appropriate element of X_u to a value larger than
!   control%infinity.
!
!  X is a rank-one array of dimension n and type default
!   real, that holds the vector of the primal variables, x.
!   The j-th component of X, j = 1, ... , n, contains (x)_j.
!
!  R is a rank-one array of dimension m and type default
!   real, that holds the vector of residuals Ao x - b.
!   The i-th component of R, i = 1, ... , m, contains (Ao x - b)_i.
!
!  Z is a rank-one array of dimension n and type default
!   real, that holds the vector of the dual variables, z.
!   The j-th component of Z, j = 1, ... , n, contains (z)_j.
!
!  X_stat is a rank-one array of dimension n and type default integer,
!   that mwill be set on exit to indicate which constraints are in the final
!   working set. Possible exit values are
!   X_stat( i ) < 0, the i-th bound constraint is in the working set,
!                    on its lower bound,
!               > 0, the i-th bound constraint is in the working set
!                    on its upper bound, and
!               = 0, the i-th bound constraint is not in the working set
!
!  regularization_weight is an optional scalar of type default real that
!   may be set to the value of the non-negative regularization weight.
!   If it is absent, the regularization weight will be zero.
!
!  W is an optional rank-one array of type default real that may be
!   set to the values of the components of the weights W.
!   The i-th component of W, i = 1, ... , o, contains (w)_i.
!   If it is absent, the weights will all be taken to be 1.0.

     INTEGER ( KIND = ip_ ), INTENT( OUT ) :: status
     TYPE ( BLLSB_full_data_type ), INTENT( INOUT ) :: data
     REAL ( KIND = rp_ ), DIMENSION( : ), INTENT( IN ) :: Ao_val
     REAL ( KIND = rp_ ), DIMENSION( : ), INTENT( IN ) :: B
     REAL ( KIND = rp_ ), DIMENSION( : ), INTENT( IN ) :: X_l, X_u
     REAL ( KIND = rp_ ), DIMENSION( : ), INTENT( INOUT ) :: X, Z
     REAL ( KIND = rp_ ), DIMENSION( : ), INTENT( OUT ) :: R
     INTEGER ( KIND = ip_ ), DIMENSION( : ), INTENT( OUT ) :: X_stat
     REAL ( KIND = rp_ ), OPTIONAL, INTENT( IN ) :: regularization_weight
     REAL ( KIND = rp_ ), OPTIONAL, INTENT( IN ), DIMENSION( data%prob%o ) :: W

!  local variables

     INTEGER ( KIND = ip_ ) :: n, o

!  recover the dimensions

     n = data%prob%n ; o = data%prob%o

!  save the observations

     data%prob%B( : o ) = B( : o )

!  save the lower and upper simple bounds

     data%prob%X_l( : n ) = X_l( : n )
     data%prob%X_u( : n ) = X_u( : n )

!  save the initial primal and dual variables and Lagrange multipliers

     data%prob%X( : n ) = X( : n )
     data%prob%Z( : n ) = Z( : n )

!  save the objective design matrix Ao entries

     IF ( data%prob%Ao%ne > 0 )                                                &
       data%prob%Ao%val( : data%prob%Ao%ne ) = Ao_val( : data%prob%Ao%ne )

!  call the solver

     CALL BLLSB_solve( data%prob, data%bllsb_data, data%bllsb_control,         &
                       data%bllsb_inform, regularization_weight, W )

!  recover the optimal primal and dual variables, Lagrange multipliers,
!  constraint values and status values for constraints and simple bounds

     X( : n ) = data%prob%X( : n )
     Z( : n ) = data%prob%Z( : n )
     R( : o ) = data%prob%R( : o )
     X_stat( : n ) = data%prob%X_status( : n )

     status = data%bllsb_inform%status
     RETURN

!  End of subroutine BLLSB_solve_blls

     END SUBROUTINE BLLSB_solve_blls

!-  G A L A H A D -  B L L S B _ i n f o r m a t i o n   S U B R O U T I N E  -

     SUBROUTINE BLLSB_information( data, inform, status )

!  return solver information during or after solution by BLLSB
!  See BLLSB_solve for a description of the required arguments

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     TYPE ( BLLSB_full_data_type ), INTENT( INOUT ) :: data
     TYPE ( BLLSB_inform_type ), INTENT( OUT ) :: inform
     INTEGER ( KIND = ip_ ), INTENT( OUT ) :: status

!  recover inform from internal data

     inform = data%bllsb_inform

!  flag a successful call

     status = GALAHAD_ok
     RETURN

!  end of subroutine BLLSB_information

     END SUBROUTINE BLLSB_information

!  End of module BLLSB

    END MODULE GALAHAD_BLLSB_precision
