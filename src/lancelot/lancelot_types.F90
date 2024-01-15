! THIS VERSION: GALAHAD 4.2 - 2023-12-13 AT 11:10 GMT.

#ifdef LANCELOT_USE_MA57
#define SILS_control MA57_control
#define SILS_factors MA57_factors
#define SILS_ainfo MA57_ainfo
#define SILS_finfo MA57_finfo
#define SILS_sinfo MA57_sinfo
#define SILS_data MA57_data
#define SILS_cntl MA57_cntl
#endif

#define IS_LANCELOT_MODULE 1
#include "galahad_modules.h"

!!$ #ifdef LANCELOT_USE_MA57
!!$ #define GALAHAD_SILS_double HSL_MA57_double
!!$ #define GALAHAD_SILS_single HSL_MA57_single
!!$ #endif

!-*-*-  L A N C E L O T  -B-  L A N C E L O T _ T Y P E S _  M O D U L E  -*-*-

!  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
!  Principal author: Nick Gould

!  History -
!   extracted from LANCELOT B. March 12th 2014

!  For full documentation, see
!   http://galahad.rl.ac.uk/galahad-www/specs.html

   MODULE LANCELOT_TYPES_precision

!  |-------------------------------------------------------|
!  |                                                       |
!  |  Derived types used by LANCELOT B and its associates  |
!  |                                                       |
!  |-------------------------------------------------------|

     USE GALAHAD_KINDS_precision
     USE GALAHAD_EXTEND_precision, ONLY: EXTEND_save_type
     USE LANCELOT_ASMBL_precision, ONLY: ASMBL_save_type
     USE LANCELOT_CAUCHY_precision, ONLY: CAUCHY_save_type
     USE LANCELOT_CG_precision, ONLY: CG_save_type
     USE LANCELOT_OTHERS_precision, ONLY: OTHERS_fdgrad_save_type
     USE LANCELOT_PRECN_precision, ONLY: PRECN_save_type
     USE GALAHAD_SCU_precision, ONLY: SCU_matrix_type, SCU_data_type,          &
                                      SCU_inform_type
     USE GALAHAD_SILS_precision, ONLY: SILS_control, SILS_factors,             &
                                       SILS_ainfo, SILS_finfo, SILS_sinfo
     USE GALAHAD_SMT_precision, ONLY: SMT_type
     IMPLICIT NONE

     PRIVATE
     PUBLIC :: LANCELOT_problem_type, LANCELOT_save_type, LANCELOT_data_type,  &
               LANCELOT_control_type, LANCELOT_inform_type

!  Set other parameters

     REAL ( KIND = rp_ ), PARAMETER :: zero = 0.0_rp_
     REAL ( KIND = rp_ ), PARAMETER :: one = 1.0_rp_
     REAL ( KIND = rp_ ), PARAMETER :: ten = 10.0_rp_

!  ======================================
!  The LANCELOT_problem_type derived type
!  ======================================

     TYPE :: LANCELOT_problem_type
       INTEGER ( KIND = ip_ ) :: n, ng, nel
       INTEGER ( KIND = ip_ ), ALLOCATABLE, DIMENSION( : ) :: IELING
       INTEGER ( KIND = ip_ ), ALLOCATABLE, DIMENSION( : ) :: ISTADG
       INTEGER ( KIND = ip_ ), ALLOCATABLE, DIMENSION( : ) :: IELVAR
       INTEGER ( KIND = ip_ ), ALLOCATABLE, DIMENSION( : ) :: ISTAEV
       INTEGER ( KIND = ip_ ), ALLOCATABLE, DIMENSION( : ) :: INTVAR
       INTEGER ( KIND = ip_ ), ALLOCATABLE, DIMENSION( : ) :: ISTADH
       INTEGER ( KIND = ip_ ), ALLOCATABLE, DIMENSION( : ) :: ICNA
       INTEGER ( KIND = ip_ ), ALLOCATABLE, DIMENSION( : ) :: ISTADA
       INTEGER ( KIND = ip_ ), ALLOCATABLE, DIMENSION( : ) :: KNDOFG
       INTEGER ( KIND = ip_ ), ALLOCATABLE, DIMENSION( : ) :: ITYPEE
       INTEGER ( KIND = ip_ ), ALLOCATABLE, DIMENSION( : ) :: ISTEPA
       INTEGER ( KIND = ip_ ), ALLOCATABLE, DIMENSION( : ) :: ITYPEG
       INTEGER ( KIND = ip_ ), ALLOCATABLE, DIMENSION( : ) :: ISTGPA
       REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : ) :: A
       REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : ) :: B
       REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : ) :: BL
       REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : ) :: BU
       REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : ) :: X
       REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : ) :: C
       REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : ) :: Y
       REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : ) :: GSCALE
       REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : ) :: ESCALE
       REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : ) :: VSCALE
       REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : ) :: EPVALU
       REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : ) :: GPVALU
       LOGICAL, ALLOCATABLE, DIMENSION( : ) :: GXEQX
       LOGICAL, ALLOCATABLE, DIMENSION( : ) :: INTREP
       CHARACTER ( LEN = 10 ), ALLOCATABLE, DIMENSION( : ) :: VNAMES
       CHARACTER ( LEN = 10 ), ALLOCATABLE, DIMENSION( : ) :: GNAMES
     END TYPE LANCELOT_problem_type

!  ===================================
!  The LANCELOT_save_type derived type
!  ===================================

     TYPE :: LANCELOT_save_type

       LOGICAL :: full_solution

!  the relevant parts of the old INWKSP common block

       INTEGER ( KIND = ip_ ) :: igetfd
       LOGICAL :: unsucc

!  the old AUGLG saved variables

       INTEGER ( KIND = ip_ ) :: nobjgr, m, icrit, ncrit, p_type
       REAL ( KIND = rp_ ) :: ocnorm, cnorm_major, etak, eta0, omegak, omega0
       REAL ( KIND = rp_ ) :: tau, tau_steering, gamma1, alphae, betae, alphak
       REAL ( KIND = rp_ ) :: alphao, betao, omega_min, eta_min, epstol, epsgrd
       REAL ( KIND = rp_ ) :: cnorm
       CHARACTER ( LEN = 5 ), DIMENSION( 5 ) :: STATE
       LOGICAL :: itzero, reeval

!  the old SBMIN saved variables

       INTEGER ( KIND = ip_ ) :: ifactr, ldx, lfxi, lgxi, lhxi, lggfx, nvar2
       INTEGER ( KIND = ip_ ) :: nfreef, nfree, nnonnz, nadd, icfact
       INTEGER ( KIND = ip_ ) :: jumpto, nbprod, infor, number, nfixed, ibqpst
       INTEGER ( KIND = ip_ ) :: nmhist, maxsel, ntotin, nfreec, lnguvl, lnhuvl
       INTEGER ( KIND = ip_ ) :: ntype, nsets, nvargp, l_suc, msweep, nbnd
       INTEGER ( KIND = ip_ ) :: mortor_its, ntotel,inform_status
       INTEGER ( KIND = ip_ ) :: nvrels, nnza, error, out, print_level
       INTEGER ( KIND = ip_ ) :: start_print, stop_print, print_gap
       INTEGER ( KIND = ip_ ) :: n_steering, n_steering_this_iteration
       INTEGER ( KIND = ip_ ) :: first_derivatives, second_derivatives
       REAL ( KIND = rp_ ) :: epstlp, gmodel, vscmax, rad, maximum_radius
       REAL ( KIND = rp_ ) :: epsrcg, fnew, radmin, cgstop, diamin, diamax
       REAL ( KIND = rp_ ) :: ared, prered, rho, fmodel, curv, dxsqr, fcp, f0
       REAL ( KIND = rp_ ) :: stepmx, smallh, resmin, qgnorm, oldrad, epscns
       REAL ( KIND = rp_ ) :: radtol, fill, step, teneps, stpmin, epstln
       REAL ( KIND = rp_ ) :: f_min, f_r, f_c, sigma_r, sigma_c, findmx
       REAL ( KIND = rp_ ) :: f_min_lag, f_r_lag, f_c_lag
       REAL ( KIND = rp_ ) :: f_min_viol, f_r_viol, f_c_viol
       REAL ( KIND = rp_ ) :: violation, delta_qv, delta_qv_steering
       LOGICAL :: alllin, altriv, next, second, print_header, modchl
       LOGICAL :: iprcnd, munks , seprec, densep, calcdi, xactcp, reusec
       LOGICAL :: gmpspr, slvbqp, refact, fdgrad, centrl, dprcnd, strctr
       LOGICAL :: use_band, icfs, mortor, firsup, twonrm, direct, myprec
       LOGICAL :: prcond, firstc, nobnds, getders, save_c
       LOGICAL :: printt, printi, printm, printw, printd, printe, set_printe
       LOGICAL :: set_printt, set_printi, set_printm, set_printw, set_printd
       LOGICAL :: skipg, steering, new_major
       CHARACTER ( LEN = 6 ) :: cgend, lisend
       CHARACTER ( LEN = 1 ) :: cgend1, lisend1
       REAL ( KIND = KIND( 1.0E0 ) ) :: t, time, tmv, tca, tls, tup
       INTEGER ( KIND = ip_ ), DIMENSION( 5 ) :: ISYS
       CHARACTER ( LEN = 6 ), DIMENSION( 6 ) :: CGENDS
       CHARACTER ( LEN = 6 ), DIMENSION( 5 ) :: LSENDS
       CHARACTER ( LEN = 1 ), DIMENSION( 6 ) :: CGENDS1
       CHARACTER ( LEN = 1 ), DIMENSION( 5 ) :: LSENDS1

!  the old CAUCH saved variables

       TYPE( CAUCHY_save_type ) :: CAUCHY

!  the old CG saved variables

       TYPE( CG_save_type ) :: CG

!  the old ASMBL saved variables

       TYPE( ASMBL_save_type ) :: ASMBL

!  the old PRECN saved variables

       TYPE( PRECN_save_type ) :: PRECN

!  the old OTHERS saved variables

       TYPE( OTHERS_fdgrad_save_type ) :: OTHERS

!  the old EXTEND saved variables

       TYPE( EXTEND_save_type ) :: EXTEND

     END TYPE LANCELOT_save_type

!  ===================================
!  The LANCELOT_data_type derived type
!  ===================================

     TYPE :: LANCELOT_data_type

       TYPE( LANCELOT_save_type ) :: S

       INTEGER ( KIND = ip_ ), ALLOCATABLE, DIMENSION( : ) :: ITRANS
       INTEGER ( KIND = ip_ ), ALLOCATABLE, DIMENSION( : ) :: ROW_start
       INTEGER ( KIND = ip_ ), ALLOCATABLE, DIMENSION( : ) :: POS_in_H
       INTEGER ( KIND = ip_ ), ALLOCATABLE, DIMENSION( : ) :: USED
       INTEGER ( KIND = ip_ ), ALLOCATABLE, DIMENSION( : ) :: FILLED
       INTEGER ( KIND = ip_ ), ALLOCATABLE, DIMENSION( : ) :: LINK_elem_uses_var
       REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : ) :: WTRANS

       INTEGER ( KIND = ip_ ), ALLOCATABLE, DIMENSION( : ) :: ISYMMD
       INTEGER ( KIND = ip_ ), ALLOCATABLE, DIMENSION( : ) :: ISWKSP
       INTEGER ( KIND = ip_ ), ALLOCATABLE, DIMENSION( : ) :: ISTAJC
       INTEGER ( KIND = ip_ ), ALLOCATABLE, DIMENSION( : ) :: ISTAGV
       INTEGER ( KIND = ip_ ), ALLOCATABLE, DIMENSION( : ) :: ISVGRP
       INTEGER ( KIND = ip_ ), ALLOCATABLE, DIMENSION( : ) :: ISLGRP
       INTEGER ( KIND = ip_ ), ALLOCATABLE, DIMENSION( : ) :: IGCOLJ
       INTEGER ( KIND = ip_ ), ALLOCATABLE, DIMENSION( : ) :: IVALJR
       INTEGER ( KIND = ip_ ), ALLOCATABLE, DIMENSION( : ) :: IUSED
       INTEGER ( KIND = ip_ ), ALLOCATABLE, DIMENSION( : ) :: ITYPER
       INTEGER ( KIND = ip_ ), ALLOCATABLE, DIMENSION( : ) :: ISSWTR
       INTEGER ( KIND = ip_ ), ALLOCATABLE, DIMENSION( : ) :: ISSITR
       INTEGER ( KIND = ip_ ), ALLOCATABLE, DIMENSION( : ) :: ISET
       INTEGER ( KIND = ip_ ), ALLOCATABLE, DIMENSION( : ) :: ISVSET
       INTEGER ( KIND = ip_ ), ALLOCATABLE, DIMENSION( : ) :: INVSET
       INTEGER ( KIND = ip_ ), ALLOCATABLE, DIMENSION( : ) :: IFREE
       INTEGER ( KIND = ip_ ), ALLOCATABLE, DIMENSION( : ) :: INDEX
       INTEGER ( KIND = ip_ ), ALLOCATABLE, DIMENSION( : ) :: IFREEC
       INTEGER ( KIND = ip_ ), ALLOCATABLE, DIMENSION( : ) :: INNONZ
       INTEGER ( KIND = ip_ ), ALLOCATABLE, DIMENSION( : ) :: LIST_elements
       INTEGER ( KIND = ip_ ), ALLOCATABLE, DIMENSION( : , : ) :: ISYMMH
       REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : ) :: FUVALS_temp
       REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : ) :: P
       REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : ) :: X0
       REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : ) :: XCP
       REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : ) :: GX0
       REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : ) :: RADII
       REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : ) :: DELTAX
       REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : ) :: QGRAD
       REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : ) :: GRJAC
       REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : ) :: CDASH
       REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : ) :: C2DASH
       REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : ) :: GV_old
       REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : , : ) :: BND
       REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : , : ) :: BND_radius

       INTEGER ( KIND = ip_ ), ALLOCATABLE, DIMENSION( : ) :: IW_asmbl
       INTEGER ( KIND = ip_ ), ALLOCATABLE, DIMENSION( : ) :: NZ_comp_w
       REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : ) :: W_ws
       REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : ) :: W_el
       REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : ) :: W_in
       REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : ) :: H_el
       REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : ) :: H_in

       INTEGER ( KIND = ip_ ), ALLOCATABLE, DIMENSION( : , : ) :: IKEEP
       INTEGER ( KIND = ip_ ), ALLOCATABLE, DIMENSION( : , : ) :: IW1
       INTEGER ( KIND = ip_ ), ALLOCATABLE, DIMENSION( : ) :: IW
       INTEGER ( KIND = ip_ ), ALLOCATABLE, DIMENSION( : ) :: IVUSE
       INTEGER ( KIND = ip_ ), ALLOCATABLE, DIMENSION( : ) :: H_col_ptr
       INTEGER ( KIND = ip_ ), ALLOCATABLE, DIMENSION( : ) :: L_col_ptr
       REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : ) :: W
       REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : ) :: RHS
       REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : ) :: RHS2
       REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : ) :: P2
       REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : ) :: G
       REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : ) :: DIAG
       REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : ) :: BREAKP
       REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : ) :: GRAD
       REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : , : ) :: W1
       REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : , : ) :: OFFDIA

       REAL ( KIND = rp_ ), POINTER, DIMENSION( : ) :: GROUP_SCALING => NULL( )
       LOGICAL, POINTER, DIMENSION( : ) :: GXEQX_AUG => NULL( )

       TYPE ( SCU_matrix_type ) :: SCU_matrix
       TYPE ( SCU_data_type ) :: SCU_data
       TYPE ( SMT_type ) :: matrix
       TYPE ( SILS_factors ) :: SILS_data

     END TYPE LANCELOT_data_type

!  ======================================
!  The LANCELOT_control_type derived type
!  ======================================

     TYPE :: LANCELOT_control_type

!  Error and ordinary output unit numbers

       INTEGER ( KIND = ip_ ) :: error = 6
       INTEGER ( KIND = ip_ ) :: out = 6

!  Removal of the file alive_file from unit alive_unit causes execution
!  to cease

       INTEGER ( KIND = ip_ ) :: alive_unit = 60
       CHARACTER ( LEN = 30 ) :: alive_file = 'ALIVE.d                       '

!  Level of output required. <= 0 gives no output, = 1 gives a one-line
!  summary for every iteration, = 2 gives a summary of the inner iteration
!  for each iteration, >= 3 gives increasingly verbose (debugging) output

       INTEGER ( KIND = ip_ ) :: print_level = 0

!  Maximum number of iterations

       INTEGER ( KIND = ip_ ) :: maxit = 1000

!   Any printing will start on this iteration (-1 = always print)

       INTEGER ( KIND = ip_ ) :: start_print = - 1

!   Any printing will stop on this iteration (-1 = always print)

       INTEGER ( KIND = ip_ ) :: stop_print = - 1

!   Printing will only occur every print_gap iterations

       INTEGER ( KIND = ip_ ) :: print_gap = 1

!  linear_solver gives the method to be used for solving the
!                linear system. 1=CG, 2=diagonal preconditioned CG,
!                3=user-provided preconditioned CG, 4=expanding band
!                preconditioned CG, 5=Munksgaard's preconditioned CG,
!                6=Schnabel-Eskow modified Cholesky preconditioned CG,
!                7=Gill-Murray-Ponceleon-Saunders modified Cholesky
!                preconditioned CG, 8=band matrix preconditioned CG,
!                9=Lin-More' preconditioned CG, 11=multifrontal direct
!                method, 12=direct modified multifrontal method

       INTEGER ( KIND = ip_ ) :: linear_solver = 8

!  The number of vectors allowed in Lin and More's incomplete factorization

       INTEGER ( KIND = ip_ ) :: icfact = 5

!  The semi-bandwidth of the band factorization

       INTEGER ( KIND = ip_ ) :: semibandwidth = 5

!   The maximum dimension of the Schur complement

       INTEGER ( KIND = ip_ ) :: max_sc = 100

!  Unit number of i/o buffer for writing temporary files (if needed)

       INTEGER ( KIND = ip_ ) :: io_buffer = 75

!  more_toraldo >= 1 gives the number of More'-Toraldo projected searches
!                to be used to improve upon the Cauchy point, anything
!                else is for the standard add-one-at-a-time CG search

       INTEGER ( KIND = ip_ ) :: more_toraldo = 0

!   non-monotone <= 0 monotone strategy used, anything else non-monotone
!                strategy with this history length used

       INTEGER ( KIND = ip_ ) :: non_monotone = 1

!  first_derivatives = 0 if exact first derivatives are given, = 1 if forward
!             finite difference approximations are to be calculated, and
!             = 2 if central finite difference approximations are to be used

       INTEGER ( KIND = ip_ ) :: first_derivatives = 0

!  second_derivatives specifies the approximation to the second derivatives
!                used. 0=exact, 1=BFGS, 2=DFP, 3=PSB, 4=SR1

       INTEGER ( KIND = ip_ ) :: second_derivatives = 0

!  Overall convergence tolerances. The iteration will terminate when the norm
!  of violation of the constraints (the "primal infeasibility") is smaller than
!  control%stopc and the norm of the gradient of the Lagrangian function (the
!  "dual infeasibility") is smaller than control%stopg

       REAL ( KIND = rp_ ) :: stopc = ten ** ( - 5 )
       REAL ( KIND = rp_ ) :: stopg = ten ** ( - 5 )

!  It will also terminate if the merit function (objective or augmented
!  Lagrangian as appropriate) is smaller than control%min_aug

       REAL ( KIND = rp_ ) :: min_aug = - ( HUGE( one ) / 8.0_rp_ )

!  Require a relative reduction in the resuiduals from CG of at least acccg

       REAL ( KIND = rp_ ) :: acccg = 0.01_rp_

!  The initial trust-region radius - a non-positive value allows the
!  package to choose its own

       REAL ( KIND = rp_ ) :: initial_radius = - one

!  The largest possible trust-region radius

       REAL ( KIND = rp_ ) :: maximum_radius = ten ** 20

!  Parameters that define when to decrease/increase the trust-region
!  (specialists only!)

       REAL ( KIND = rp_ ) :: eta_successful = 0.01_rp_
       REAL ( KIND = rp_ ) :: eta_very_successful = 0.9_rp_
       REAL ( KIND = rp_ ) :: eta_extremely_successful = 0.95_rp_
       REAL ( KIND = rp_ ) :: gamma_smallest = 0.0625_rp_
       REAL ( KIND = rp_ ) :: gamma_decrease = 0.25_rp_
       REAL ( KIND = rp_ ) :: gamma_increase = 2.0_rp_
       REAL ( KIND = rp_ ) :: mu_meaningful_model = 0.01_rp_
       REAL ( KIND = rp_ ) :: mu_meaningful_group = 0.1_rp_

!  The initial value of the penalty parameter

       REAL ( KIND = rp_ ) :: initial_mu = 0.1_rp_

!  The penalty parameter decrease factor

       REAL ( KIND = rp_ ) :: mu_decrease = 0.1_rp_

!  The penalty parameter decrease factor when steering

       REAL ( KIND = rp_ ) :: mu_steering_decrease = 0.7_rp_

!  The value of the penalty parameter above which the algorithm
!  will not attempt to update the estimates of the Lagrange multipliers

       REAL ( KIND = rp_ ) :: mu_tol = 0.1_rp_

!  The required accuracy of the norm of the projected gradient at the end
!  of the first major iteration

       REAL ( KIND = rp_ ) :: firstg = 0.1_rp_

!  The required accuracy of the norm of the constraints at the end
!  of the first major iteration

       REAL ( KIND = rp_ ) :: firstc = 0.1_rp_

!  control parameters from Curtis-Jiang-Robinson steering

       INTEGER ( KIND = ip_ ) :: num_mudec = HUGE( 1 )
       INTEGER ( KIND = ip_ ) :: num_mudec_per_iteration = HUGE( 1 )
       REAL ( KIND = rp_ ) :: kappa_3 = ten ** ( - 5 )
       REAL ( KIND = rp_ ) :: kappa_t = 0.9_rp_
       REAL ( KIND = rp_ ) :: mu_min = zero

!   the maximum CPU time allowed (-ve means infinite)

        REAL ( KIND = rp_ ) :: cpu_time_limit = - one

!  Is the function quadratic ?

       LOGICAL :: quadratic_problem = .FALSE.

!  Do we want to steer the iterates towards feasibility ?

       LOGICAL :: steering = .FALSE.

!  two_norm_tr is true if a 2-norm trust-region is to be used, and false
!                for the infinity norm

       LOGICAL :: two_norm_tr = .FALSE.

!  exact_gcp is true if the exact Cauchy point is required, and false if an
!                approximation suffices

       LOGICAL :: exact_gcp = .TRUE.

!  use a Gauss-Newton model of the infeasibility rather than a second-order
!   Taylor model

       LOGICAL :: gn_model = .FALSE.

!  use a Gauss-Newton model of the infeasibility rather than a second-order
!   Taylor model once the Cauchy point has been found

       LOGICAL :: gn_model_after_cauchy = .FALSE.

!  magical_steps is true if magical steps are to be used to improve upon
!                already accepted points, and false otherwise

       LOGICAL :: magical_steps = .FALSE.

!  accurate_bqp is true if the the minimizer of the quadratic model within
!                the intersection of the trust-region and feasible box
!                is to be sought (to a prescribed accuracy), and false
!                if an approximation suffices

       LOGICAL :: accurate_bqp = .FALSE.

!  structured_tr is true if a structured trust region will be used, and false
!                if a standard trust-region suffices

       LOGICAL :: structured_tr = .FALSE.

!  For printing, if we are maximizing rather than minimizing, print_max
!  should be .TRUE.

       LOGICAL :: print_max = .FALSE.

!  .TRUE. if all components of the solution and constraints are to be printed
!  on termination, and .FALSE. if only the first and last (representative) few
!  are required

       LOGICAL :: full_solution = .TRUE.

!  space_critical. If true, every effort will be made to use as little
!  space as possible. This may result in longer computation times

       LOGICAL :: space_critical = .FALSE.

!  deallocate_error_fatal. If true, any array/pointer deallocation error
!  will terminate execution. Otherwise, computation will continue

       LOGICAL :: deallocate_error_fatal = .FALSE.

!  control parameters for SILS

       TYPE ( SILS_control ) :: SILS_cntl
     END TYPE LANCELOT_control_type

!  =====================================
!  The LANCELOT_inform_type derived type
!  =====================================

     TYPE :: LANCELOT_inform_type

!  return status. See LANCELOT_solve for details

       INTEGER ( KIND = ip_ ) :: status = 0

!  the status of the last attempted allocation/deallocation

       INTEGER ( KIND = ip_ ) :: alloc_status = 0

!  the total number of iterations required

       INTEGER ( KIND = ip_ ) :: iter = - 1

!  the total number of CG iterations required

       INTEGER ( KIND = ip_ ) :: itercg = - 1

!  the maximum number of CG iterations permitted per inner iteration

       INTEGER ( KIND = ip_ ) :: itcgmx = - 1

!  the number of element functions that must be re-evaluated when %status > 0

       INTEGER ( KIND = ip_ ) :: ncalcf = 0

!  the number of group functions that must be re-evaluated when %status > 0

       INTEGER ( KIND = ip_ ) :: ncalcg = 0

!  the current number of free variables

       INTEGER ( KIND = ip_ ) :: nvar = 0

!  the number of derivative evaluations made

       INTEGER ( KIND = ip_ ) :: ngeval = 0

!  the total number of secant updates that are skipped

       INTEGER ( KIND = ip_ ) :: iskip = 0

!  the variable that most recently encountered on of its bounds

       INTEGER ( KIND = ip_ ) :: ifixed = 0

!  the bandwidth used with the expanding-band preconditioner

       INTEGER ( KIND = ip_ ) :: nsemib = 0

!  the value of the augmented Lagrangian merit function at the best estimate
!   of the solution determined by LANCELOT_solve

       REAL ( KIND = rp_ ) :: aug = HUGE( one )

!  the value of the objective function at the best estimate of the solution
!   determined by LANCELOT_solve

       REAL ( KIND = rp_ ) :: obj = HUGE( one )

!  the norm of the projected gradient of the merit function

       REAL ( KIND = rp_ ) :: pjgnrm = HUGE( one )

!  the infinity norm of the equality constraints

       REAL ( KIND = rp_ ) :: cnorm = zero

!  the current ratio of predicted to achieved merit function reduction

       REAL ( KIND = rp_ ) :: ratio = zero

!  the current value of the penalty parameter

       REAL ( KIND = rp_ ) :: mu = zero

!  the current value of the trust-region radius

       REAL ( KIND = rp_ ) :: radius = zero

!  the pivot tolerance used when ICCG is used for preconditioning

       REAL ( KIND = rp_ ) :: ciccg = zero

!  newsol is true if a major iteration has just been completed

       LOGICAL :: newsol = .FALSE.

!  the name of the array for which an allocation error occurred

       CHARACTER ( LEN = 80 ) :: bad_alloc =  REPEAT( ' ', 80 )

!  return information from SCU

       TYPE ( SCU_inform_type ) :: SCU_info

!  return information from SILS

       TYPE ( SILS_ainfo ) :: SILS_infoa
       TYPE ( SILS_finfo ) :: SILS_infof
       TYPE ( SILS_sinfo ) :: SILS_infos
     END TYPE LANCELOT_inform_type

!  End of module LANCELOT_types

   END MODULE LANCELOT_TYPES_precision
