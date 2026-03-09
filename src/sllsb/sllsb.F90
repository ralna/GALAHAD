! THIS VERSION: GALAHAD 5.5 - 2026-01-29 AT 13:00 GMT.

#include "galahad_modules.h"

!-*-*-*-*-*-*-*-*-*-  G A L A H A D _ S L L S B    M O D U L E  -*-*-*-*-*-*-*-

!  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
!  Principal author: Nick Gould

!  History -
!   originally released with GALAHAD  Version 5.5, January 19th 2026

!  For full documentation, see
!   http://galahad.rl.ac.uk/galahad-www/specs.html

    MODULE GALAHAD_SLLSB_precision

!      -------------------------------------------------------
!     | Minimize the least-squares objective function         |
!     |                                                       |
!     |  1/2 || A_o x - b ||_W^2 + 1/2 weight || x - x_s ||^2 |
!     |                                                       |
!     | subject to the non-overlapping simplex constraints    |
!     |                                                       |
!     |      e_Ci^T x_Ci = 1, x_Ci >= 0, i = 1,..., m,        |
!     |                                                       |
!     | using an infeasible-point primal-dual method          |
!      -------------------------------------------------------

!  ** This is essentially a wrapper for GALAHAD_CLLS with A set to hold
!  the simplex constraints

      USE GALAHAD_KINDS_precision
!$    USE omp_lib
      USE GALAHAD_CLLS_precision, SLLSB_control_type => CLLS_control_type,     &
                                  SLLSB_time_type => CLLS_time_type,           &
                                  SLLSB_inform_type => CLLS_inform_type
      USE GALAHAD_CLOCK
      USE GALAHAD_SYMBOLS
      USE GALAHAD_STRING, ONLY: STRING_pleural, STRING_verb_pleural,           &
                                STRING_ies, STRING_are, STRING_ordinal
      USE GALAHAD_SPACE_precision
      USE GALAHAD_SMT_precision
      USE GALAHAD_QPT_precision
      USE GALAHAD_SPECFILE_precision
      USE GALAHAD_LSP_precision, SLLSB_dims_type => QPT_dimensions_type
      USE GALAHAD_QPD_precision, SLLSB_data_type => QPD_data_type,             &
                                 SLLSB_AX => QPD_AX,                           &
                                 SLLSB_abs_AX => QPD_abs_AX,                   &
                                 SLLSB_AoX => QPD_A_by_col_X,                  &
                                 SLLSB_abs_AoX => QPD_abs_A_by_col_X
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
      PUBLIC :: SLLSB_initialize, SLLSB_read_specfile, SLLSB_solve,            &
                SLLSB_terminate, SLLSB_control_type, SLLSB_data_type,          &
                SLLSB_time_type, SLLSB_inform_type, SLLSB_information,         &
                SLLSB_full_initialize, SLLSB_full_terminate,                   &
                SLLSB_import, SLLSB_solve_given_a, SLLSB_reset_control,        &
                QPT_problem_type, SMT_type, SMT_put, SMT_get

!----------------------
!   I n t e r f a c e s
!----------------------

     INTERFACE SLLSB_initialize
       MODULE PROCEDURE SLLSB_initialize, SLLSB_full_initialize
     END INTERFACE SLLSB_initialize

     INTERFACE SLLSB_terminate
       MODULE PROCEDURE SLLSB_terminate, SLLSB_full_terminate
     END INTERFACE SLLSB_terminate

!----------------------
!   P a r a m e t e r s
!----------------------

      REAL ( KIND = rp_ ), PARAMETER :: zero = 0.0_rp_
      REAL ( KIND = rp_ ), PARAMETER :: one = 1.0_rp_

!-------------------------------------------------
!  D e r i v e d   t y p e   d e f i n i t i o n s
!-------------------------------------------------

      TYPE, PUBLIC :: SLLSB_full_data_type
        LOGICAL :: f_indexing = .TRUE.
        TYPE ( SLLSB_data_type ) :: SLLSB_data
        TYPE ( SLLSB_control_type ) :: SLLSB_control
        TYPE ( SLLSB_inform_type ) :: SLLSB_inform
        TYPE ( QPT_problem_type ) :: prob
      END TYPE SLLSB_full_data_type

   CONTAINS

!-*-*-*-*-   S L L S B _ I N I T I A L I Z E   S U B R O U T I N E   -*-*-*-*-

      SUBROUTINE SLLSB_initialize( data, control, inform )

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
!
!  Default control data for SLLSB. This routine should be called before
!  SLLSB_solve
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

      TYPE ( SLLSB_data_type ), INTENT( INOUT ) :: data
      TYPE ( SLLSB_control_type ), INTENT( OUT ) :: control
      TYPE ( SLLSB_inform_type ), INTENT( OUT ) :: inform

      CALL CLLS_initialize( data, control, inform )
      RETURN

!  End of SLLSB_initialize

      END SUBROUTINE SLLSB_initialize

! G A L A H A D - S L L S B _ F U L L _ I N I T I A L I Z E  S U B R O U T I N E

     SUBROUTINE SLLSB_full_initialize( data, control, inform )

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!   Provide default values for SLLSB controls

!   Arguments:

!   data     private internal data
!   control  a structure containing control information. See preamble
!   inform   a structure containing output information. See preamble

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     TYPE ( SLLSB_full_data_type ), INTENT( INOUT ) :: data
     TYPE ( SLLSB_control_type ), INTENT( OUT ) :: control
     TYPE ( SLLSB_inform_type ), INTENT( OUT ) :: inform

     CALL SLLSB_initialize( data%sllsb_data, control, inform )

     RETURN

!  End of subroutine SLLSB_full_initialize

     END SUBROUTINE SLLSB_full_initialize

!-*-*-*-   S L L S B _ R E A D _ S P E C F I L E  S U B R O U T I N E   -*-*-*-

      SUBROUTINE SLLSB_read_specfile( control, device, alt_specname )

!  Reads the content of a specification file, and performs the assignment of
!  values associated with given keywords to the corresponding control parameters

!  The defauly values as given by SLLSB_initialize could (roughly)
!  have been set as:

! BEGIN SLLSB SPECIFICATIONS (DEFAULT)
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
!  sif-file-name                                     SLLSBPROB.SIF
!  qplib-file-name                                   SLLSBPROB.qplib
!  output-line-prefix                                ""
! END SLLSB SPECIFICATIONS (DEFAULT)

!  Dummy arguments

      TYPE ( SLLSB_control_type ), INTENT( INOUT ) :: control
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
      CHARACTER( LEN = 5 ), PARAMETER :: specname = 'SLLSB'
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

!     IF ( PRESENT( alt_specname ) ) WRITE(6,*) ' sllsb: ', alt_specname

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

      END SUBROUTINE SLLSB_read_specfile

!-*-*-*-*-*-*-*-*-   S L L S B _ S O L V E  S U B R O U T I N E   -*-*-*-*-*-*-

      SUBROUTINE SLLSB_solve( prob, data, control, inform )

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
!
!  Minimize the linear least-squares objective function
!
!         1/2 || A_o x - b ||_W^2 + 1/2 weight || x ||^2
!
!  subject to the non-overlapping simplex constraints
!
!          e_Ci^T x_Ci = 1, x_Ci >= 0, i = 1,..., m,
!
!  x is a vector of n components ( x_1, .... , x_n ),  A_o is an o by n
!  matrix, the weighted norm ||v||_W = sqrt( sum_i=1^o w_i v_i^2 ), and
!  the Ci are non-overlapping index subsets of {1,...,n} using a primal-dual 
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
!   %new_problem_structure is a LOGICAL variable, that must be set to
!    .TRUE. by the user if this is the first problem with this "structure"
!    to be solved since the last call to SLLSB_initialize, and .FALSE. if
!    a previous call to a problem with the same "structure" (but different
!    numerical data) was made.
!
!   %n is an INTEGER variable, that must be set by the user to the
!    number of optimization parameters, n.  RESTRICTION: %n >= 1
!
!   %o is an INTEGER variable, that must be set by the user to the
!    number of observations, o.  RESTRICTION: %o >= 1
!
!   %m is an INTEGER variable, that should be set by the user to the
!    number of simplex constraints, m, invoved. RESTRICTION: %m >= 0
!
!   %regularization_weight is a REAL variable, that may be set by the user
!    to the value of the non-negative regularization weight. It takes the 
!    default value of zero
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
!   %B is a REAL array of length %o, that must be set by the user to the value
!    of the observations, b. The i-th component of B, i = 1, ...., %o should
!    contain the value of b_i.
!
!   %COHORT is a INTEGER array of length %n, whose j-th component may be set to
!    the number, between 1 and %m, of the cohort to which variable x_j belongs,
!    or to 0 if the variable belong to no cohort. If COHORT is unallocated,
!    all variables will be assumed to belong to a single cohort.
!
!   %W is a REAL array of length %o, that may be set by the user
!    to the values of the components of the weights w.
!    If %W is unallocated, the weights will all be taken to be 1.0.
!
!   %X_s is a REAL array of length %n, that may be set by the user
!    to the values of the components of the shifts x_s. 
!    If %X_s is unallocated, the shifts will all be taken to be 0.0.
!
!   %X is a REAL array of length %n, that must be set by the user
!    to estimaes of the solution, x. On successful exit, it will contain
!    the required solution, x.
!
!   %Z is a REAL array of length %n, that must be set by the user to
!    appropriate estimates of the values of the dual variables
!    (Lagrange multipliers corresponding to the simple bound constraints
!    x >= 0). On successful exit, it will contain the required vector of 
!    dual variables.
!
!   %R is a REAL array of length %o, that is used to store the values of
!    the residuals A_o x - b. It need not be set on entry. On exit, it will
!    have been filled with appropriate values.
!
!   %X_status is an INTEGER array of length %n, that will be set on exit to
!    indicate the likely ultimate status of the non-negativity constraints
!    Possible values are
!    X_status( i ) < 0, the i-th bound constraint is likely in the active set,
!                       on its lower bound, and
!                  = 0, the i-th bound constraint is likely not in the active
!                       set
!    It need not be set on entry.
!
!  data is a structure of type SLLSB_data_type that holds private internal data
!
!  control is a structure of type SLLSB_control_type that controls the
!   execution of the subroutine and must be set by the user. Default values for
!   the elements may be set by a call to SLLSB_initialize. See the preamble
!   for details
!
!  inform is a structure of type SLLSB_inform_type that provides
!    information on exit from SLLSB_solve. The component status
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
!  On exit from SLLSB_solve, other components of inform are given in the
!   preamble to CLLS
!
! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!  Dummy arguments

      TYPE ( QPT_problem_type ), INTENT( INOUT ) :: prob
      TYPE ( SLLSB_data_type ), INTENT( INOUT ) :: data
      TYPE ( SLLSB_control_type ), INTENT( IN ) :: control
      TYPE ( SLLSB_inform_type ), INTENT( OUT ) :: inform

!  Local variables

      INTEGER ( KIND = ip_ ) :: i, j, k, l, status, alloc_status
      REAL :: time_start, time_now
      REAL ( KIND = rp_ ) :: clock_start, clock_now
      CHARACTER ( LEN = 80 ) :: array_name, bad_alloc

!  initialize time

      CALL CPU_TIME( time_start ) ; CALL CLOCK_time( clock_start )

!  if there is possibly more than one cohort, set and store the number of 
!  variables occurring in each cohort in prob%A%ptr

      IF ( ALLOCATED( prob%COHORT ) .AND. prob%m >= 1 ) THEN
        array_name = 'sllsb: prob%A%ptr'
        CALL SPACE_resize_array( prob%m + 1, prob%A%ptr, inform%status,        &
               inform%alloc_status, array_name = array_name,                   &
               deallocate_error_fatal = control%deallocate_error_fatal,        &
               exact_size = control%space_critical,                            &
               bad_alloc = inform%bad_alloc, out = control%error )
        IF ( inform%status /= GALAHAD_ok ) GO TO 910

!  store the counts

        prob%A%ne = 0 ; prob%A%ptr( 1 : prob%m ) = 0
        DO j = 1, prob%n
          i = prob%COHORT( j )
          IF ( i > prob%m .OR. i < 0 ) THEN
            inform%status = GALAHAD_error_restrictions ; GO TO 910
          ELSE IF ( i > 0 ) THEN
            prob%A%ne = prob%A%ne + 1
            prob%A%ptr( i ) = prob%A%ptr( i ) + 1
          END IF
        END DO

!  check to make sure each cohort is non empty

        IF ( MINVAL( prob%A%ptr( 1 : prob%m ) ) == 0 ) THEN
          inform%status = GALAHAD_error_restrictions ; GO TO 910
        END IF

!  otherwise there is only one cohort covering all variables

      ELSE
        prob%m = 1 ; prob%A%ne = prob%n
        array_name = 'sllsb: prob%A%ptr'
        CALL SPACE_resize_array( prob%m + 1, prob%A%ptr, inform%status,        &
               inform%alloc_status, array_name = array_name,                   &
               deallocate_error_fatal = control%deallocate_error_fatal,        &
               exact_size = control%space_critical,                            &
               bad_alloc = inform%bad_alloc, out = control%error )
        IF ( inform%status /= GALAHAD_ok ) GO TO 910
      END IF

!  set space for the cohort matrix stored by rows, as well as lower and upper
!  bounds on the constraints and variables

      prob%A%m = prob%m ; prob%A%n = prob%n

      CALL SMT_put( prob%A%type, 'SPARSE_BY_ROWS', inform%alloc_status )
      IF ( inform%alloc_status /= 0 ) THEN
        inform%status = GALAHAD_error_allocate ; GO TO 910
      END IF

      array_name = 'sllsb: prob%A%col'
      CALL SPACE_resize_array( prob%A%ne, prob%A%col, inform%status,           &
             inform%alloc_status, array_name = array_name,                     &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 910

      array_name = 'sllsb: prob%A%val'
      CALL SPACE_resize_array( prob%A%ne, prob%A%val, inform%status,           &
             inform%alloc_status, array_name = array_name,                     &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 910

      array_name = 'sllsb: prob%C'
      CALL SPACE_resize_array( prob%m, prob%C, inform%status,                  &
             inform%alloc_status, array_name = array_name,                     &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 910

      array_name = 'sllsb: prob%C_l'
      CALL SPACE_resize_array( prob%m, prob%C_l, inform%status,                &
             inform%alloc_status, array_name = array_name,                     &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 910

      array_name = 'sllsb: prob%C_u'
      CALL SPACE_resize_array( prob%m, prob%C_u, inform%status,                &
             inform%alloc_status, array_name = array_name,                     &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 910

      array_name = 'sllsb: prob%Y'
      CALL SPACE_resize_array( prob%m, prob%Y, inform%status,                  &
             inform%alloc_status, array_name = array_name,                     &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 910

      array_name = 'sllsb: prob%X_l'
      CALL SPACE_resize_array( prob%n, prob%X_l, inform%status,                &
             inform%alloc_status, array_name = array_name,                     &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 910

      array_name = 'sllsb: prob%X_u'
      CALL SPACE_resize_array( prob%n, prob%X_u, inform%status,                &
             inform%alloc_status, array_name = array_name,                     &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 910

      array_name = 'sllsb: prob%Z'
      CALL SPACE_resize_array( prob%n, prob%Z, inform%status,                  &
             inform%alloc_status, array_name = array_name,                     &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 910

!  record that the simplex constraints require that the values in each cohort
!  sum to one

      prob%C_l( 1 : prob%m ) = one
      prob%C_u( 1 : prob%m ) = one

!  record the cohort matrix by rows

      IF ( ALLOCATED( prob%COHORT ) .AND. prob%m >= 1 ) THEN

!  store the first entry in cohort i in position prob%A%ptr(i)

        k = 1
        DO i = 1, prob%m
          l = prob%A%ptr( i )
          prob%A%ptr( i ) = k
          k = k + l
        END DO

!  next march through the cohorts, inserting those in cohort i consecutively in
!  prob%A%col/val, as well as setting appropriate lower and upper bounds on x

        DO j = 1, prob%n
          i = prob%COHORT( j )
          IF ( i > 0 ) THEN
            l = prob%A%ptr( i )
            prob%A%col( l ) = j
            prob%A%val( l ) = one
            prob%X_l( j ) = zero
            prob%X_u( j ) = control%infinity
            prob%A%ptr( i ) = l + 1
          ELSE
            prob%X_l( j ) = - control%infinity
            prob%X_u( j ) = control%infinity
          END IF
        END DO     

!  finally, reset the pointers to the first entries in each row

        DO i = prob%m, 1, - 1
          prob%A%ptr( i + 1 ) = prob%A%ptr( i )
        END DO
        prob%A%ptr( 1 ) = 1

!  there is only one cohort

      ELSE
        prob%m = 1
        prob%A%ne = prob%n
        prob%A%ptr( 1 ) = 1
        prob%A%ptr( prob%m + 1 ) = prob%n + 1
        prob%A%col( 1 : prob%n ) = [ ( i, i = 1, prob%n ) ]
        prob%A%val( 1 : prob%n ) = one
        prob%X_l( 1 : prob%n ) = zero
        prob%X_u( 1 : prob%n ) = control%infinity
      END IF

!     write( 6, * ) ' n, o, m ', prob%n, prob%o, prob%m
!     write( 6, * ) ' A ptr ', prob%A%ptr
!     write( 6, * ) ' A col ', prob%A%col
!     write( 6, * ) ' A val ', prob%A%val
!     write( 6, * ) ' X_l ', prob%X_l
!     write( 6, * ) ' X_u ', prob%X_u
!     write( 6, * ) ' C_l ', prob%C_l
!     write( 6, * ) ' C_u ', prob%C_u

!  now call the generic interior-point constrained linear least-squares solver

      CALL CLLS_solve( prob, data, control, inform )

!  save status values while arrays are deallocated

      status = inform%status
      alloc_status = inform%alloc_status
      bad_alloc = inform%bad_alloc

!  deallocate space used for null A and its associates

      array_name = 'sllsb: prob%A%col'
      CALL SPACE_dealloc_array( prob%A%col,                                    &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'sllsb: prob%A%ptr'
      CALL SPACE_dealloc_array( prob%A%ptr,                                    &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'sllsb: prob%A%val'
      CALL SPACE_dealloc_array( prob%A%val,                                    &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'sllsb: prob%A%type'
      CALL SPACE_dealloc_array( prob%A%type,                                   &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'sllsb: prob%C'
      CALL SPACE_dealloc_array( prob%C,                                        &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'sllsb: prob%C_l'
      CALL SPACE_dealloc_array( prob%C_l,                                      &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'sllsb: prob%C_u'
      CALL SPACE_dealloc_array( prob%C_u,                                      &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

!     array_name = 'sllsb: prob%Y'
!     CALL SPACE_dealloc_array( prob%Y,                                        &
!        inform%status, inform%alloc_status, array_name = array_name,          &
!        bad_alloc = inform%bad_alloc, out = control%error )
!     IF ( control%deallocate_error_fatal .AND.                                &
!          inform%status /= GALAHAD_ok ) RETURN

      array_name = 'sllsb: prob%X_l'
      CALL SPACE_dealloc_array( prob%X_l,                                      &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'sllsb: prob%X_u'
      CALL SPACE_dealloc_array( prob%X_u,                                      &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

!     array_name = 'sllsb: prob%Z'
!     CALL SPACE_dealloc_array( prob%Z,                                        &
!        inform%status, inform%alloc_status, array_name = array_name,          &
!        bad_alloc = inform%bad_alloc, out = control%error )
!     IF ( control%deallocate_error_fatal .AND.                                &
!          inform%status /= GALAHAD_ok ) RETURN

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

!  end of SUBROUTINE SLLSB_solve

      END SUBROUTINE SLLSB_solve

! -*-*-*-*-*-   S L L S B _ T E R M I N A T E   S U B R O U T I N E   -*-*-*-*-

      SUBROUTINE SLLSB_terminate( data, control, inform )

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!      ..............................................
!      .                                            .
!      .  Deallocate internal arrays at the end     .
!      .  of the computation                        .
!      .                                            .
!      ..............................................

!  Arguments:
!
!   data    see Subroutine SLLSB_initialize
!   control see Subroutine SLLSB_initialize
!   inform  see Subroutine SLLSB_solve

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!  Dummy arguments

      TYPE ( SLLSB_data_type ), INTENT( INOUT ) :: data
      TYPE ( SLLSB_control_type ), INTENT( IN ) :: control
      TYPE ( SLLSB_inform_type ), INTENT( INOUT ) :: inform

      CALL CLLS_terminate( data, control, inform )
      RETURN

!  End of subroutine SLLSB_terminate

      END SUBROUTINE SLLSB_terminate

!  G A L A H A D -  S L L S B _ f u l l _ t e r m i n a t e  S U B R O U T I N E

     SUBROUTINE SLLSB_full_terminate( data, control, inform )

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!   Deallocate all private storage

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     TYPE ( SLLSB_full_data_type ), INTENT( INOUT ) :: data
     TYPE ( SLLSB_control_type ), INTENT( IN ) :: control
     TYPE ( SLLSB_inform_type ), INTENT( INOUT ) :: inform

!-----------------------------------------------
!   L o c a l   V a r i a b l e s
!-----------------------------------------------

     CHARACTER ( LEN = 80 ) :: array_name

!  deallocate workspace

     CALL SLLSB_terminate( data%sllsb_data, control, inform )

!  deallocate any internal problem arrays

     array_name = 'sllsb: data%prob%X'
     CALL SPACE_dealloc_array( data%prob%X,                                    &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'slls: data%prob%Y'
     CALL SPACE_dealloc_array( data%prob%Y,                                    &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'sllsb: data%prob%Z'
     CALL SPACE_dealloc_array( data%prob%Z,                                    &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'slls: data%prob%COHORT'
     CALL SPACE_dealloc_array( data%prob%COHORT,                               &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'sllsb: data%prob%R'
     CALL SPACE_dealloc_array( data%prob%R,                                    &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'sllsb: data%prob%X_l'
     CALL SPACE_dealloc_array( data%prob%X_l,                                  &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'sllsb: data%prob%X_u'
     CALL SPACE_dealloc_array( data%prob%X_u,                                  &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'sllsb: data%prob%C_l'
     CALL SPACE_dealloc_array( data%prob%C_l,                                  &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'sllsb: data%prob%C_u'
     CALL SPACE_dealloc_array( data%prob%C_u,                                  &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'sllsb: data%prob%Ao%ptr'
     CALL SPACE_dealloc_array( data%prob%Ao%ptr,                               &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'sllsb: data%prob%Ao%row'
     CALL SPACE_dealloc_array( data%prob%Ao%row,                               &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'sllsb: data%prob%Ao%col'
     CALL SPACE_dealloc_array( data%prob%Ao%col,                               &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'sllsb: data%prob%Ao%val'
     CALL SPACE_dealloc_array( data%prob%Ao%val,                               &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'sllsb: data%prob%Ao%type'
     CALL SPACE_dealloc_array( data%prob%Ao%type,                              &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'sllsb: data%prob%X_status'
     CALL SPACE_dealloc_array( data%prob%X_status,                             &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     RETURN

!  End of subroutine SLLSB_full_terminate

     END SUBROUTINE SLLSB_full_terminate

! -----------------------------------------------------------------------------
! =============================================================================
! -----------------------------------------------------------------------------
!              specific interfaces to make calls from C easier
! -----------------------------------------------------------------------------
! =============================================================================
! -----------------------------------------------------------------------------

!-*-*-*-  G A L A H A D -  S L L S B _ i m p o r t _ S U B R O U T I N E -*-*-*-

     SUBROUTINE SLLSB_import( control, data, status, n, o, m,                  &
                              Ao_type, Ao_ne, Ao_row, Ao_col, Ao_ptr,          &
                              COHORT )

!  import fixed problem data into internal storage prior to solution.
!  Arguments are as follows:

!  control is a derived type whose components are described in the leading
!   comments to SLLSB_solve
!
!  data is a scalar variable of type SLLSB_full_data_type used for internal data
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
!  m is a scalar variable of type default integer, that holds the number of 
!   cohorts
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
!  COHORT is an optional rank-one array of type default integer and length n
!   that must be set so that its j-th component is a number, between 1 and m, 
!   of the cohort to which variable x_j belongs, or to 0 if the variable 
!   belong to no cohort. If m or COHORT is absent, all variables will be 
!   assumed to belong to a single cohort
!
!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     TYPE ( SLLSB_control_type ), INTENT( INOUT ) :: control
     TYPE ( SLLSB_full_data_type ), INTENT( INOUT ) :: data
     INTEGER ( KIND = ip_ ), INTENT( IN ) :: n, o, m, Ao_ne
     INTEGER ( KIND = ip_ ), INTENT( OUT ) :: status
     CHARACTER ( LEN = * ), INTENT( IN ) :: Ao_type
     INTEGER ( KIND = ip_ ), DIMENSION( : ), OPTIONAL, INTENT( IN ) :: Ao_row
     INTEGER ( KIND = ip_ ), DIMENSION( : ), OPTIONAL, INTENT( IN ) :: Ao_col
     INTEGER ( KIND = ip_ ), DIMENSION( : ), OPTIONAL, INTENT( IN ) :: Ao_ptr
     INTEGER ( KIND = ip_ ), DIMENSION( n ), OPTIONAL, INTENT( IN ) :: COHORT

!  local variables

     INTEGER ( KIND = ip_ ) :: error
     LOGICAL :: deallocate_error_fatal, space_critical
     CHARACTER ( LEN = 80 ) :: array_name

!  copy control to data

     WRITE( control%out, "( '' )", ADVANCE = 'no') ! prevents ifort bug
     data%sllsb_control = control

     error = data%sllsb_control%error
     space_critical = data%sllsb_control%space_critical
     deallocate_error_fatal = data%sllsb_control%space_critical

!  if there are multiple cohorts, record them

     IF ( PRESENT( COHORT ) ) THEN
       data%prob%m = m
       array_name = 'slls: data%prob%COHORT'
       CALL SPACE_resize_array( n, data%prob%COHORT,                           &
              data%sllsb_inform%status, data%sllsb_inform%alloc_status,        &
              array_name = array_name,                                         &
              deallocate_error_fatal = deallocate_error_fatal,                 &
              exact_size = space_critical,                                     &
              bad_alloc = data%sllsb_inform%bad_alloc, out = error )
       IF ( data%sllsb_inform%status /= 0 ) GO TO 900
       
       IF ( data%f_indexing ) THEN
         data%prob%COHORT( : n ) = MAX( COHORT( : n ), 0 )
       ELSE
         data%prob%COHORT( : n ) = MAX( COHORT( : n ) + 1, 0 )
       END IF
     ELSE
       data%prob%m = 1
     END IF

!  allocate vector space if required

     array_name = 'sllsb: data%prob%B'
     CALL SPACE_resize_array( o, data%prob%B,                                  &
            data%sllsb_inform%status, data%sllsb_inform%alloc_status,          &
            array_name = array_name,                                           &
            deallocate_error_fatal =                                           &
              data%sllsb_control%deallocate_error_fatal,                       &
            exact_size = data%sllsb_control%space_critical,                    &
            bad_alloc = data%sllsb_inform%bad_alloc,                           &
            out = data%sllsb_control%error )
     IF ( data%sllsb_inform%status /= 0 ) GO TO 900

     array_name = 'sllsb: data%prob%X_l'
     CALL SPACE_resize_array( n, data%prob%X_l,                                &
            data%sllsb_inform%status, data%sllsb_inform%alloc_status,          &
            array_name = array_name,                                           &
            deallocate_error_fatal = deallocate_error_fatal,                   &
            exact_size = space_critical,                                       &
            bad_alloc = data%sllsb_inform%bad_alloc, out = error )
     IF ( data%sllsb_inform%status /= 0 ) GO TO 900

     array_name = 'sllsb: data%prob%X_u'
     CALL SPACE_resize_array( n, data%prob%X_u,                                &
            data%sllsb_inform%status, data%sllsb_inform%alloc_status,          &
            array_name = array_name,                                           &
            deallocate_error_fatal = deallocate_error_fatal,                   &
            exact_size = space_critical,                                       &
            bad_alloc = data%sllsb_inform%bad_alloc, out = error )
     IF ( data%sllsb_inform%status /= 0 ) GO TO 900

     array_name = 'sllsb: data%prob%X'
     CALL SPACE_resize_array( n, data%prob%X,                                  &
            data%sllsb_inform%status, data%sllsb_inform%alloc_status,          &
            array_name = array_name,                                           &
            deallocate_error_fatal = deallocate_error_fatal,                   &
            exact_size = space_critical,                                       &
            bad_alloc = data%sllsb_inform%bad_alloc, out = error )
     IF ( data%sllsb_inform%status /= 0 ) GO TO 900

     array_name = 'sllsb: data%prob%R'
     CALL SPACE_resize_array( n, data%prob%R,                                  &
            data%sllsb_inform%status, data%sllsb_inform%alloc_status,          &
            array_name = array_name,                                           &
            deallocate_error_fatal = deallocate_error_fatal,                   &
            exact_size = space_critical,                                       &
            bad_alloc = data%sllsb_inform%bad_alloc, out = error )
     IF ( data%sllsb_inform%status /= 0 ) GO TO 900

     array_name = 'sllsb: data%prob%Y'
     CALL SPACE_resize_array( data%prob%m, data%prob%Y,                        &
            data%sllsb_inform%status, data%sllsb_inform%alloc_status,          &
            array_name = array_name,                                           &
            deallocate_error_fatal = deallocate_error_fatal,                   &
            exact_size = space_critical,                                       &
            bad_alloc = data%sllsb_inform%bad_alloc, out = error )
     IF ( data%sllsb_inform%status /= 0 ) GO TO 900

     array_name = 'sllsb: data%prob%Z'
     CALL SPACE_resize_array( n, data%prob%Z,                                  &
            data%sllsb_inform%status, data%sllsb_inform%alloc_status,          &
            array_name = array_name,                                           &
            deallocate_error_fatal = deallocate_error_fatal,                   &
            exact_size = space_critical,                                       &
            bad_alloc = data%sllsb_inform%bad_alloc, out = error )
     IF ( data%sllsb_inform%status /= 0 ) GO TO 900

     array_name = 'sllsb: data%prob%X_status'
     CALL SPACE_resize_array( n, data%prob%X_status,                           &
            data%sllsb_inform%status, data%sllsb_inform%alloc_status,          &
            array_name = array_name,                                           &
            deallocate_error_fatal = deallocate_error_fatal,                   &
            exact_size = space_critical,                                       &
            bad_alloc = data%sllsb_inform%bad_alloc, out = error )
     IF ( data%sllsb_inform%status /= 0 ) GO TO 900

!  put data into the required components of the qpt storage type

     data%prob%n = n ; data%prob%o = o

!  set Ao appropriately in the qpt storage type

     SELECT CASE ( Ao_type )
     CASE ( 'coordinate', 'COORDINATE' )
       IF ( .NOT. ( PRESENT( Ao_row ) .AND. PRESENT( Ao_col ) ) ) THEN
         data%sllsb_inform%status = GALAHAD_error_optional
         GO TO 900
       END IF
       CALL SMT_put( data%prob%Ao%type, 'COORDINATE',                          &
                     data%sllsb_inform%alloc_status )
       data%prob%Ao%n = n ; data%prob%Ao%m = o
       data%prob%Ao%ne = Ao_ne

       array_name = 'sllsb: data%prob%Ao%row'
       CALL SPACE_resize_array( data%prob%Ao%ne, data%prob%Ao%row,             &
              data%sllsb_inform%status, data%sllsb_inform%alloc_status,        &
              array_name = array_name,                                         &
              deallocate_error_fatal = deallocate_error_fatal,                 &
              exact_size = space_critical,                                     &
              bad_alloc = data%sllsb_inform%bad_alloc, out = error )
       IF ( data%sllsb_inform%status /= 0 ) GO TO 900

       array_name = 'sllsb: data%prob%Ao%col'
       CALL SPACE_resize_array( data%prob%Ao%ne, data%prob%Ao%col,             &
              data%sllsb_inform%status, data%sllsb_inform%alloc_status,        &
              array_name = array_name,                                         &
              deallocate_error_fatal = deallocate_error_fatal,                 &
              exact_size = space_critical,                                     &
              bad_alloc = data%sllsb_inform%bad_alloc, out = error )
       IF ( data%sllsb_inform%status /= 0 ) GO TO 900

       array_name = 'sllsb: data%prob%Ao%val'
       CALL SPACE_resize_array( data%prob%Ao%ne, data%prob%Ao%val,             &
              data%sllsb_inform%status, data%sllsb_inform%alloc_status,        &
              array_name = array_name,                                         &
              deallocate_error_fatal = deallocate_error_fatal,                 &
              exact_size = space_critical,                                     &
              bad_alloc = data%sllsb_inform%bad_alloc, out = error )
       IF ( data%sllsb_inform%status /= 0 ) GO TO 900
       IF ( data%f_indexing ) THEN
         data%prob%Ao%row( : data%prob%Ao%ne ) = Ao_row( : data%prob%Ao%ne )
         data%prob%Ao%col( : data%prob%Ao%ne ) = Ao_col( : data%prob%Ao%ne )
       ELSE
         data%prob%Ao%row( : data%prob%Ao%ne ) = Ao_row( : data%prob%Ao%ne ) + 1
         data%prob%Ao%col( : data%prob%Ao%ne ) = Ao_col( : data%prob%Ao%ne ) + 1
       END IF

     CASE ( 'sparse_by_rows', 'SPARSE_BY_ROWS' )
       IF ( .NOT. ( PRESENT( Ao_ptr ) .AND. PRESENT( Ao_col ) ) ) THEN
         data%sllsb_inform%status = GALAHAD_error_optional
         GO TO 900
       END IF
       CALL SMT_put( data%prob%Ao%type, 'SPARSE_BY_ROWS',                      &
                     data%sllsb_inform%alloc_status )
       data%prob%Ao%n = n ; data%prob%Ao%m = o
       IF ( data%f_indexing ) THEN
         data%prob%Ao%ne = Ao_ptr( o + 1 ) - 1
       ELSE
         data%prob%Ao%ne = Ao_ptr( o + 1 )
       END IF
       array_name = 'sllsb: data%prob%Ao%ptr'
       CALL SPACE_resize_array( o + 1, data%prob%Ao%ptr,                       &
              data%sllsb_inform%status, data%sllsb_inform%alloc_status,        &
              array_name = array_name,                                         &
              deallocate_error_fatal = deallocate_error_fatal,                 &
              exact_size = space_critical,                                     &
              bad_alloc = data%sllsb_inform%bad_alloc, out = error )
       IF ( data%sllsb_inform%status /= 0 ) GO TO 900

       array_name = 'sllsb: data%prob%Ao%col'
       CALL SPACE_resize_array( data%prob%Ao%ne, data%prob%Ao%col,             &
              data%sllsb_inform%status, data%sllsb_inform%alloc_status,        &
              array_name = array_name,                                         &
              deallocate_error_fatal = deallocate_error_fatal,                 &
              exact_size = space_critical,                                     &
              bad_alloc = data%sllsb_inform%bad_alloc, out = error )
       IF ( data%sllsb_inform%status /= 0 ) GO TO 900

       array_name = 'sllsb: data%prob%Ao%val'
       CALL SPACE_resize_array( data%prob%Ao%ne, data%prob%Ao%val,             &
              data%sllsb_inform%status, data%sllsb_inform%alloc_status,        &
              array_name = array_name,                                         &
              deallocate_error_fatal = deallocate_error_fatal,                 &
              exact_size = space_critical,                                     &
              bad_alloc = data%sllsb_inform%bad_alloc, out = error )
       IF ( data%sllsb_inform%status /= 0 ) GO TO 900

       IF ( data%f_indexing ) THEN
         data%prob%Ao%ptr( : o + 1 ) = Ao_ptr( : o + 1 )
         data%prob%Ao%col( : data%prob%Ao%ne ) = Ao_col( : data%prob%Ao%ne )
       ELSE
         data%prob%Ao%ptr( : o + 1 ) = Ao_ptr( : o + 1 ) + 1
         data%prob%Ao%col( : data%prob%Ao%ne ) = Ao_col( : data%prob%Ao%ne ) + 1
       END IF

     CASE ( 'sparse_by_columns', 'SPARSE_BY_COLUMNS' )
       IF ( .NOT. ( PRESENT( Ao_ptr ) .AND. PRESENT( Ao_row ) ) ) THEN
         data%sllsb_inform%status = GALAHAD_error_optional
         GO TO 900
       END IF
       CALL SMT_put( data%prob%Ao%type, 'SPARSE_BY_COLUMNS',                   &
                     data%sllsb_inform%alloc_status )
       data%prob%Ao%n = n ; data%prob%Ao%m = o
       IF ( data%f_indexing ) THEN
         data%prob%Ao%ne = Ao_ptr( n + 1 ) - 1
       ELSE
         data%prob%Ao%ne = Ao_ptr( n + 1 )
       END IF
       array_name = 'sllsb: data%prob%Ao%ptr'
       CALL SPACE_resize_array( n + 1, data%prob%Ao%ptr,                       &
              data%sllsb_inform%status, data%sllsb_inform%alloc_status,        &
              array_name = array_name,                                         &
              deallocate_error_fatal = deallocate_error_fatal,                 &
              exact_size = space_critical,                                     &
              bad_alloc = data%sllsb_inform%bad_alloc, out = error )
       IF ( data%sllsb_inform%status /= 0 ) GO TO 900

       array_name = 'sllsb: data%prob%Ao%row'
       CALL SPACE_resize_array( data%prob%Ao%ne, data%prob%Ao%row,             &
              data%sllsb_inform%status, data%sllsb_inform%alloc_status,        &
              array_name = array_name,                                         &
              deallocate_error_fatal = deallocate_error_fatal,                 &
              exact_size = space_critical,                                     &
              bad_alloc = data%sllsb_inform%bad_alloc, out = error )
       IF ( data%sllsb_inform%status /= 0 ) GO TO 900

       array_name = 'sllsb: data%prob%Ao%val'
       CALL SPACE_resize_array( data%prob%Ao%ne, data%prob%Ao%val,             &
              data%sllsb_inform%status, data%sllsb_inform%alloc_status,        &
              array_name = array_name,                                         &
              deallocate_error_fatal = deallocate_error_fatal,                 &
              exact_size = space_critical,                                     &
              bad_alloc = data%sllsb_inform%bad_alloc, out = error )
       IF ( data%sllsb_inform%status /= 0 ) GO TO 900

       IF ( data%f_indexing ) THEN
         data%prob%Ao%ptr( : n + 1 ) = Ao_ptr( : n + 1 )
         data%prob%Ao%row( : data%prob%Ao%ne ) = Ao_row( : data%prob%Ao%ne )
       ELSE
         data%prob%Ao%ptr( : n + 1 ) = Ao_ptr( : n + 1 ) + 1
         data%prob%Ao%row( : data%prob%Ao%ne ) = Ao_row( : data%prob%Ao%ne ) + 1
       END IF

     CASE ( 'dense', 'DENSE', 'dense_by_rows', 'DENSE_BY_ROWS' )
       CALL SMT_put( data%prob%Ao%type, 'DENSE',                               &
                     data%sllsb_inform%alloc_status )
       data%prob%Ao%n = n ; data%prob%Ao%m = o
       data%prob%Ao%ne = o * n

       array_name = 'sllsb: data%prob%Ao%val'
       CALL SPACE_resize_array( data%prob%Ao%ne, data%prob%Ao%val,             &
              data%sllsb_inform%status, data%sllsb_inform%alloc_status,        &
              array_name = array_name,                                         &
              deallocate_error_fatal = deallocate_error_fatal,                 &
              exact_size = space_critical,                                     &
              bad_alloc = data%sllsb_inform%bad_alloc, out = error )
       IF ( data%sllsb_inform%status /= 0 ) GO TO 900

     CASE ( 'dense_by_columns', 'DENSE_BY_COLUMNS' )
       CALL SMT_put( data%prob%Ao%type, 'DENSE_BY_COLUMNS',                    &
                     data%sllsb_inform%alloc_status )
       data%prob%Ao%n = n ; data%prob%Ao%m = o
       data%prob%Ao%ne = o * n

       array_name = 'sllsb: data%prob%Ao%val'
       CALL SPACE_resize_array( data%prob%Ao%ne, data%prob%Ao%val,             &
              data%sllsb_inform%status, data%sllsb_inform%alloc_status,        &
              array_name = array_name,                                         &
              deallocate_error_fatal = deallocate_error_fatal,                 &
              exact_size = space_critical,                                     &
              bad_alloc = data%sllsb_inform%bad_alloc, out = error )
       IF ( data%sllsb_inform%status /= 0 ) GO TO 900

     CASE DEFAULT
       data%sllsb_inform%status = GALAHAD_error_unknown_storage
       GO TO 900
     END SELECT

     status = GALAHAD_ok
     RETURN

!  error returns

 900 CONTINUE
     status = data%sllsb_inform%status
     RETURN

!  End of subroutine SLLSB_import

     END SUBROUTINE SLLSB_import

!-  G A L A H A D -  S L L S B _ r e s e t _ c o n t r o l   S U B R O U T I N E

     SUBROUTINE SLLSB_reset_control( control, data, status )

!  reset control parameters after import if required.
!  See SLLSB_solve for a description of the required arguments

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     TYPE ( SLLSB_control_type ), INTENT( IN ) :: control
     TYPE ( SLLSB_full_data_type ), INTENT( INOUT ) :: data
     INTEGER ( KIND = ip_ ), INTENT( OUT ) :: status

!  set control in internal data

     data%sllsb_control = control

!  flag a successful call

     status = GALAHAD_ok
     RETURN

!  end of subroutine SLLSB_reset_control

     END SUBROUTINE SLLSB_reset_control

!- G A L A H A D -  S L L S B _ s o l v e _ g i v e n _ a  S U B R O U T I N E -

     SUBROUTINE SLLSB_solve_given_a( data, status, Ao_val, B,                  &
                                     regularization_weight,                    &
                                     X, Y, Z, R, X_stat, W, X_s )

!  solve the constrained linear least-squares problem whose structure was
!  previously imported. See SLLSB_solve for a description of the required
!  arguments.

!--------------------------------
!   D u m m y   A r g u m e n t s
!--------------------------------

!  data is a scalar variable of type SLLSB_full_data_type used for internal data
!
!  status is a scalar variable of type default intege that indicates the
!   success or otherwise of the import. If status = 0, the solve was succesful.
!   For other values see, sllsb_solve above.
!
!  Ao_val is a rank-one array of type default real, that holds the values
!   of the design matrix Ao in the storage scheme specified in sllsb_import.
!
!  B is a rank-one array of dimension o and type default
!   real, that holds the vector of linear terms of the observations, b.
!   The i-th component of B, i = 1, ... , o, contains (b)_i.
!
!  regularization_weight is an optional scalar of type default real that
!   holds the value of the non-negative regularization weight, sigma.
!
!  X is a rank-one array of dimension n and type default
!   real, that holds the vector of the primal variables, x.
!   The j-th component of X, j = 1, ... , n, contains (x)_j.
!
!  Y is a rank-one array of dimension m and type default
!   real, that holds the vector of the Lagrange multipliers, y.
!   The i-th component of Y, i = 1, ... , m, contains (y)_i.
!
!  Z is a rank-one array of dimension n and type default
!   real, that holds the vector of the dual variables, z.
!   The j-th component of Z, j = 1, ... , n, contains (z)_j.
!
!  R is a rank-one array of dimension m and type default
!   real, that holds the vector of residuals Ao x - b.
!   The i-th component of R, i = 1, ... , m, contains (Ao x - b)_i.
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
!  W is an optional rank-one array of type default real that may be
!   set to the values of the components of the weights W.
!   The i-th component of W, i = 1, ... , o, contains (w)_i.
!   If it is absent, the weights will all be taken to be 1.0.
!
!  X_s is an optional rank-one array of type default real that may be
!   set to the values of the components of the shifts X_s.
!   The j-th component of X_s, j = 1, ... , n, contains (x_s)_j.
!   If it is absent, the shifts will all be taken to be 0.0.

     INTEGER ( KIND = ip_ ), INTENT( OUT ) :: status
     TYPE ( SLLSB_full_data_type ), INTENT( INOUT ) :: data
     REAL ( KIND = rp_ ), INTENT( IN ) :: regularization_weight
     REAL ( KIND = rp_ ), DIMENSION( : ), INTENT( IN ) :: Ao_val
     REAL ( KIND = rp_ ), DIMENSION( : ), INTENT( IN ) :: B
     REAL ( KIND = rp_ ), DIMENSION( : ), INTENT( INOUT ) :: X, Y, Z
     REAL ( KIND = rp_ ), DIMENSION( : ), INTENT( OUT ) :: R
     INTEGER ( KIND = ip_ ), DIMENSION( : ), INTENT( OUT ) :: X_stat
     REAL ( KIND = rp_ ), INTENT( IN ), OPTIONAL, DIMENSION( data%prob%o ) :: W
     REAL ( KIND = rp_ ), INTENT( IN ), OPTIONAL,                              &
                                        DIMENSION( data%prob%n ) :: X_s

!  local variables

     INTEGER ( KIND = ip_ ) :: n, o, m, error
     LOGICAL :: deallocate_error_fatal, space_critical, w_present, x_s_present
     CHARACTER ( LEN = 80 ) :: array_name

!  record whether W and X_s are present

     w_present = PRESENT ( W ) ;  x_s_present = PRESENT ( X_s )

!  recover the dimensions

     n = data%prob%n ; o = data%prob%o ; m = data%prob%m

!  save the regularization weight

     data%prob%regularization_weight = regularization_weight

!  save the observations

     data%prob%B( : o ) = B( : o )

!  save the initial primal and dual variables and Lagrange multipliers

     data%prob%X( : n ) = X( : n )
     data%prob%Y( : m ) = Y( : m )
     data%prob%Z( : n ) = Z( : n )

!  save the objective design matrix Ao entries

     IF ( data%prob%Ao%ne > 0 )                                                &
       data%prob%Ao%val( : data%prob%Ao%ne ) = Ao_val( : data%prob%Ao%ne )

!  save the weights if they are present

     IF ( PRESENT( W ) ) THEN
       array_name = 'sllsb: data%prob%W'
       CALL SPACE_resize_array( o, data%prob%W,                                &
              data%sllsb_inform%status, data%sllsb_inform%alloc_status,        &
              array_name = array_name,                                         &
              deallocate_error_fatal = deallocate_error_fatal,                 &
              exact_size = space_critical,                                     &
            bad_alloc = data%sllsb_inform%bad_alloc, out = error )
       IF ( data%sllsb_inform%status /= 0 ) GO TO 900
       data%prob%W( : o ) = W( : o )
     END IF

!  save the shifts if they are present

     IF ( PRESENT( X_s ) ) THEN
       array_name = 'sllsb: data%prob%X_s'
       CALL SPACE_resize_array( n, data%prob%X_s,                              &
              data%sllsb_inform%status, data%sllsb_inform%alloc_status,        &
              array_name = array_name,                                         &
              deallocate_error_fatal = deallocate_error_fatal,                 &
              exact_size = space_critical,                                     &
            bad_alloc = data%sllsb_inform%bad_alloc, out = error )
       IF ( data%sllsb_inform%status /= 0 ) GO TO 900
       data%prob%X_s( : n ) = X_s( : n )
     END IF

!  call the solver

     CALL SLLSB_solve( data%prob, data%sllsb_data, data%sllsb_control,         &
                       data%sllsb_inform )

!  recover the optimal primal and dual variables, Lagrange multipliers,
!  residual values and status values for the simple bounds

     IF ( SYMBOLS_success( data%sllsb_inform%status ) ) THEN
       X( : n ) = data%prob%X( : n )
       Y( : m ) = data%prob%Y( : m )
       Z( : n ) = data%prob%Z( : n )
       R( : o ) = data%prob%R( : o )
       X_stat( : n ) = data%prob%X_status( : n )
     END IF

     status = data%sllsb_inform%status
     RETURN

!  error returns

 900 CONTINUE
     status = data%sllsb_inform%status
     RETURN

!  End of subroutine SLLSB_solve_given_a

     END SUBROUTINE SLLSB_solve_given_a

!-  G A L A H A D -  S L L S B _ i n f o r m a t i o n   S U B R O U T I N E  -

     SUBROUTINE SLLSB_information( data, inform, status )

!  return solver information during or after solution by SLLSB
!  See SLLSB_solve for a description of the required arguments

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     TYPE ( SLLSB_full_data_type ), INTENT( INOUT ) :: data
     TYPE ( SLLSB_inform_type ), INTENT( OUT ) :: inform
     INTEGER ( KIND = ip_ ), INTENT( OUT ) :: status

!  recover inform from internal data

     inform = data%sllsb_inform

!  flag a successful call

     status = GALAHAD_ok
     RETURN

!  end of subroutine SLLSB_information

     END SUBROUTINE SLLSB_information

!  End of module SLLSB

    END MODULE GALAHAD_SLLSB_precision
