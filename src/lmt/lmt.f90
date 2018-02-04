! THIS VERSION: GALAHAD 2.6 - 15/12/2014 AT 09:15 GMT.

!-*-*-*-*-*-*-*-*-*-*- G A L A H A D _ L M T   M O D U L E -*-*-*-*-*-*-*-*-*-

!  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
!  Principal author: Nick Gould

!  History -
!   originally released as part of GALAHAD_LMS Version 2.6. June 12th 2014
!   became self-contained module, December 15th 2014

!  For full documentation, see 
!   http://galahad.rl.ac.uk/galahad-www/specs.html

    MODULE GALAHAD_LMT_double

!    ----------------------------------------------------------------
!   |                                                                |
!   | Derived types for limited-memory secant Hessian approximations |
!   |                                                                |
!    ----------------------------------------------------------------

     IMPLICIT NONE

      PRIVATE

!--------------------
!   P r e c i s i o n
!--------------------

      INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )

!-------------------------------------------------
!  D e r i v e d   t y p e   d e f i n i t i o n s
!-------------------------------------------------

!  - - - - - - - - - - - - - - - - - - - - - - - 
!   control derived type with component defaults
!  - - - - - - - - - - - - - - - - - - - - - - - 

      TYPE, PUBLIC :: LMT_control_type

!  unit for error messages

        INTEGER :: error = 6

!  unit for monitor output

        INTEGER :: out = 6

!  controls level of diagnostic output

        INTEGER :: print_level = 0

!  limited memory length

        INTEGER :: memory_length = 10

!  limited-memory method required (others may be added in due course):
!    1 BFGS (default)
!    2 SR1
!    3 BFGS inverse
!    4 shifted BFGS inverse

        INTEGER :: method = 1

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

        REAL ( KIND = wp ) :: total = 0.0

!  cpu time spent setting up space for the secant approximation

        REAL ( KIND = wp ) :: setup = 0.0

!  cpu time spent updating the secant approximation

        REAL ( KIND = wp ) :: form = 0.0

!  cpu time spent applying the secant approximation

        REAL ( KIND = wp ) :: apply = 0.0

!  total clock time spent in the package

        REAL ( KIND = wp ) :: clock_total = 0.0

!  clock time spent setting up space for the secant approximation

        REAL ( KIND = wp ) :: clock_setup = 0.0

!  clock time spent updating the secant approximation

        REAL ( KIND = wp ) :: clock_form = 0.0

!  clock time spent applying the secant approximation

        REAL ( KIND = wp ) :: clock_apply = 0.0

      END TYPE LMT_time_type

!  - - - - - - - - - - - - - - - - - - - - - - - 
!   inform derived type with component defaults
!  - - - - - - - - - - - - - - - - - - - - - - - 

      TYPE, PUBLIC :: LMT_inform_type

!  return status. See LMT_setup for details

        INTEGER :: status = 0

!  the status of the last attempted allocation/deallocation

        INTEGER :: alloc_status = 0

!  the number of pairs (s,y) currently used to represent the limited-memory 
!   matrix

        INTEGER :: length = - 1

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
        INTEGER :: n, m, latest, length, len_c, lwork, n_restriction
        INTEGER :: restricted = 0
        REAL ( KIND = wp ) :: delta, gamma, lambda, delta_plus_lambda
        REAL ( KIND = wp ) :: one_over_dpl, d_over_dpl
        LOGICAL :: full, need_form_shift, sr1_singular
        LOGICAL :: any_method = .FALSE.
        CHARACTER ( LEN = 6 ) :: method
        TYPE ( LMT_control_type ) :: control
        INTEGER, ALLOCATABLE, DIMENSION( : ) :: ORDER, PIVOTS, RESTRICTION
        REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: WORK
        REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : , : ) :: S, Y
        REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : , : ) :: YTY, YTS, STS
        REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : , : ) :: C, R, L_scaled
        REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : , : ) :: QP, QP_perm
      END TYPE LMT_data_type

!  end of module GALAHAD_LMT_double

    END MODULE GALAHAD_LMT_double



