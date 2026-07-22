! THIS VERSION: GALAHAD 5.6 - 2026-07-22 AT 13:30 GMT.

#include "galahad_modules.h"

!-*-*- G A L A H A D  -  Q P O A S E S  _ I N T E R F A C E   M O D U L E -*-*-

  MODULE QPOASES_INTERFACE_precision
    USE GALAHAD_KINDS_precision, ONLY: ipc_, rpc_

    IMPLICIT NONE

    PRIVATE

!  module to provide fortran interfaces to qpOASES C++ structures and functions

!  Copyright reserved, GALAHAD productions
!  Principal authors: Nick Gould and the Google Gemini AI

!  History -
!   originally released in GALAHAD Version 5.6. July 18th 2026

!----------------------
!   P a r a m e t e r s
!----------------------

    REAL ( KIND = rpc_ ), PARAMETER :: ten = 10.0_rpc_
    REAL ( KIND = rpc_ ), PARAMETER :: epsmch = EPSILON( ten )

!-------------------------------------------------
!  D e r i v e d   t y p e   d e f i n i t i o n s
!-------------------------------------------------

    TYPE, PUBLIC, BIND( C ) :: qpOASES_options_type

!  print level (-2=debug,-1=table,0=none,1=error,2=default,3=all)

        INTEGER ( ipc_ ) :: printLevel = 0

!  specifies whether ramping shall be enabled or not 

        INTEGER ( ipc_ ) :: enableRamping = 1

!  specifies whether far bounds shall be used or not 

        INTEGER ( ipc_ ) :: enableFarBounds = 1

!  specifies whether flipping bounds shall be used or not 

        INTEGER ( ipc_ ) :: enableFlippingBounds = 1

!  specifies whether Hessian matrix shall be regularised in case 
!  semi-definitenesS is detected 

        INTEGER ( ipc_ ) :: enableRegularisation = 0

!  specifies whether condition-hardened LI test shall be used or not 

        INTEGER ( ipc_ ) :: enableFullLITests = 0

!  specifies whether nonzero curvature tests shall be used 

        INTEGER ( ipc_ ) :: enableNZCTests = 1

!  specifies the frequency of drift corrections (0 = off) 

        INTEGER ( ipc_ ) :: enableDriftCorrection = 1

!  specifies the frequency of full refactorisation of proj Hessian 
!  (otherwise updates) 

        INTEGER ( ipc_ ) :: enableCholeskyRefactorisation = 0

!  specifies whether equalities shall be always treated as active constraints 

        INTEGER ( ipc_ ) :: enableEqualities = 0

!  termination tolerance 

#ifdef REAL_32
        REAL ( rpc_ ) :: terminationTolerance = ten ** 2 * epsmch
#else
        REAL ( rpc_ ) :: terminationTolerance = 5000000.0_rpc_ * epsmch
#endif

!  lower/upper (constraints') bound tolerance (an inequality constraint whose 
!  lower and upper bounds differ by less is regarded to be an equality 
!  constraint) 

#ifdef REAL_32
        REAL ( rpc_ ) :: boundTolerance = ten ** 2 * epsmch     
#else
        REAL ( rpc_ ) :: boundTolerance = ten ** 6 * epsmch     
#endif
                                                                                
!  offset for relaxing (constraints') bounds at beginning of an initial 
!  homotopy. It is also as initial value for far bounds 

        REAL ( rpc_ ) :: boundRelaxation = ten ** 4

!  numerator tolerance for ratio tests 

#ifdef REAL_32
        REAL ( rpc_ ) :: epsNum = - ten ** 2 * epsmch
#else
        REAL ( rpc_ ) :: epsNum = ten ** 2 * epsmch
#endif

!  denominator tolerance for ratio tests 

#ifdef REAL_32
        REAL ( rpc_ ) :: epsDen = - ten ** 3 * epsmch
#else
        REAL ( rpc_ ) :: epsDen = ten ** 3 * epsmch
#endif

!  maximum allowed jump in primal variables in nonzero curvature tests 

        REAL ( rpc_ ) :: maxPrimalJump = ten ** 8

!  maximum allowed jump in dual variables in linear independence tests 

        REAL ( rpc_ ) :: maxDualJump = ten ** 8

!  start value for Ramping Strategy 

        REAL ( rpc_ ) :: initialRamping = 0.5_rpc_

!  final value for Ramping Strategy 

        REAL ( rpc_ ) :: finalRamping = 1.0_rpc_

!  initial size of Far Bounds 

        REAL ( rpc_ ) :: initialFarBounds = ten ** 6                            

!  factor to grow Far Bounds 

        REAL ( rpc_ ) :: growFarBounds = ten ** 3                               

!  initial status of bounds at first iteration: 
!   -1=lower,0=free,1=upper,2=infeas lower,3=infeas upper,4=undefined)

        INTEGER ( ipc_ ) :: initialStatusBounds = - 1

!  tolerance of squared Cholesky diagonal factor which triggers flipping bound 

#ifdef REAL_32
        REAL ( rpc_ ) :: epsFlipping = 50.0_rpc_* epsmch
#else
        REAL ( rpc_ ) :: epsFlipping = ten ** 3 * epsmch
#endif

!  maximum number of successive regularisation steps 

        INTEGER ( ipc_ ) :: numRegularisationSteps = 0

!  scaling factor of identity matrix used for Hessian regularisation 

#ifdef REAL_32
        REAL ( rpc_ ) :: epsRegularisation = 200.0_rpc_ * epsmch
#else
        REAL ( rpc_ ) :: epsRegularisation = ten ** 3 * epsmch
#endif

!  maximum number of iterative refinement steps 

#ifdef REAL_32
        INTEGER ( ipc_ ) :: numRefinementSteps = 2
#else
        INTEGER ( ipc_ ) :: numRefinementSteps = 1
#endif

!  early termination tolerance for iterative refinement 

        REAL ( rpc_ ) :: epsIterRef = ten ** 2 * epsmch

!  tolerance for linear independence tests 

#ifdef REAL_32
        REAL ( rpc_ ) :: epsLITests = 50.0_rpc_ * epsmch
#else
        REAL ( rpc_ ) :: epsLITests = ten ** 5 * epsmch
#endif

!  tolerance for nonzero curvature tests 

#ifdef REAL_32
        REAL ( rpc_ ) :: epsNZCTests = ten ** 2 * epsmch
#else
        REAL ( rpc_ ) :: epsNZCTests = 3000.0_rpc_ * epsmch
#endif

!  minimum reciprocal condition number of S before refactorization is triggered 

        REAL ( rpc_ ) :: rcondSMin = ten ** 2 * epsmch

!  specifies whether the working set should be repaired when negative curvature
!  is discovered during hotstart 

        INTEGER ( ipc_ ) :: enableInertiaCorrection = 1
        INTEGER ( ipc_ ) :: enableDropInfeasibles = 0
        INTEGER ( ipc_ ) :: dropBoundPriority = 1
        INTEGER ( ipc_ ) :: dropEqConPriority = 1
        INTEGER ( ipc_ ) :: dropIneqConPriority = 1

!  if true, it will print the internal qpOASES residuals and other 
!  information per iteration 

        INTEGER ( ipc_ ) :: printResiduals = 0

    END TYPE qpOASES_options_type

  END MODULE QPOASES_INTERFACE_precision
