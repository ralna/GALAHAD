! THIS VERSION: GALAHAD 4.1 - 2023-01-24 AT 09:30 GMT.

#include "galahad_modules.h"

!-*-*-*-*-*-*-*-*-*-*- G A L A H A D _ L M T   M O D U L E -*-*-*-*-*-*-*-*-*-

!  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
!  Principal author: Nick Gould

!  History -
!   originally released as part of GALAHAD_LMS Version 2.6. June 12th 2014
!   became self-contained module, December 15th 2014

!  For full documentation, see
!   http://galahad.rl.ac.uk/galahad-www/specs.html

    MODULE GALAHAD_LMT_precision

!    ----------------------------------------------------------------
!   |                                                                |
!   | Derived types for limited-memory secant Hessian approximations |
!   |                                                                |
!    ----------------------------------------------------------------

      USE GALAHAD_KINDS_precision

      IMPLICIT NONE

      PRIVATE

!-------------------------------------------------
!  D e r i v e d   t y p e   d e f i n i t i o n s
!-------------------------------------------------

!  - - - - - - - - - - - - - - - - - - - - - - -
!   control derived type with component defaults
!  - - - - - - - - - - - - - - - - - - - - - - -

      TYPE, PUBLIC :: LMT_control_type

!  unit for error messages

        INTEGER ( KIND = ip_ ) :: error = 6

!  unit for monitor output

        INTEGER ( KIND = ip_ ) :: out = 6

!  controls level of diagnostic output

        INTEGER ( KIND = ip_ ) :: print_level = 0

!  limited memory length

        INTEGER ( KIND = ip_ ) :: memory_length = 10

!  limited-memory method required (others may be added in due course):
!    1 BFGS (default)
!    2 SR1
!    3 BFGS inverse
!    4 shifted BFGS inverse

        INTEGER ( KIND = ip_ ) :: method = 1

!  allow space to permit different methods if required (less efficient)

        LOGICAL :: any_method = .FALSE.

!  if space is critical, ensure allocated arrays are no bigger than needed

        LOGICAL :: space_critical = .FALSE.

!  exit if any deallocation fails

        LOGICAL :: deallocate_error_fatal  = .FALSE.

!  all output lines will be prefixed by
!    prefix(2:LEN(TRIM(%prefix))-1)
!  where prefix contains the required string enclosed in quotes,
!  e.g. "string" or 'string'

        CHARACTER ( LEN = 30 ) :: prefix = '""                            '
      END TYPE LMT_control_type

!  - - - - - - - - - - - - - - - - - - - - - -
!   time derived type with component defaults
!  - - - - - - - - - - - - - - - - - - - - - -

      TYPE, PUBLIC :: LMT_time_type

!  total cpu time spent in the package

        REAL ( KIND = rp_ ) :: total = 0.0

!  cpu time spent setting up space for the secant approximation

        REAL ( KIND = rp_ ) :: setup = 0.0

!  cpu time spent updating the secant approximation

        REAL ( KIND = rp_ ) :: form = 0.0

!  cpu time spent applying the secant approximation

        REAL ( KIND = rp_ ) :: apply = 0.0

!  total clock time spent in the package

        REAL ( KIND = rp_ ) :: clock_total = 0.0

!  clock time spent setting up space for the secant approximation

        REAL ( KIND = rp_ ) :: clock_setup = 0.0

!  clock time spent updating the secant approximation

        REAL ( KIND = rp_ ) :: clock_form = 0.0

!  clock time spent applying the secant approximation

        REAL ( KIND = rp_ ) :: clock_apply = 0.0

      END TYPE LMT_time_type

!  - - - - - - - - - - - - - - - - - - - - - - -
!   inform derived type with component defaults
!  - - - - - - - - - - - - - - - - - - - - - - -

      TYPE, PUBLIC :: LMT_inform_type

!  return status. See LMT_setup for details

        INTEGER ( KIND = ip_ ) :: status = 0

!  the status of the last attempted allocation/deallocation

        INTEGER ( KIND = ip_ ) :: alloc_status = 0

!  the number of pairs (s,y) currently used to represent the limited-memory
!   matrix

        INTEGER ( KIND = ip_ ) :: length = - 1

!  have (s,y) pairs been skipped when forming the limited-memory matrix

        LOGICAL :: updates_skipped = .FALSE.

!  the name of the array for which an allocation/deallocation error ocurred

        CHARACTER ( LEN = 80 ) :: bad_alloc = REPEAT( ' ', 80 )

!  timings (see above)

        TYPE ( LMT_time_type ) :: time
      END TYPE LMT_inform_type

!  - - - - - - - - - -
!   data derived type
!  - - - - - - - - - -

      TYPE, PUBLIC :: LMT_data_type
        INTEGER ( KIND = ip_ ) :: n, m, latest, length, len_c, lwork
        INTEGER ( KIND = ip_ ) :: n_restriction
        INTEGER ( KIND = ip_ ) :: restricted = 0
        REAL ( KIND = rp_ ) :: delta, gamma, lambda, delta_plus_lambda
        REAL ( KIND = rp_ ) :: one_over_dpl, d_over_dpl
        LOGICAL :: full, need_form_shift, sr1_singular
        LOGICAL :: any_method = .FALSE.
        CHARACTER ( LEN = 6 ) :: method
        TYPE ( LMT_control_type ) :: control
        INTEGER ( KIND = ip_ ), ALLOCATABLE, DIMENSION( : ) :: ORDER, PIVOTS
        INTEGER ( KIND = ip_ ), ALLOCATABLE, DIMENSION( : ) :: RESTRICTION
        REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : ) :: WORK
        REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : , : ) :: S, Y
        REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : , : ) :: YTY, YTS, STS
        REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : , : ) :: C, R, L_scaled
        REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : , : ) :: QP, QP_perm
      END TYPE LMT_data_type

!  end of module GALAHAD_LMT_precision

    END MODULE GALAHAD_LMT_precision



