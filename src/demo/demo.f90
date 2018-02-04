! THIS VERSION: GALAHAD 2.4 - 14/04/2009 AT 16:00 GMT.

!-*-*-*-*-*-*-*-*-  G A L A H A D _ D E M O   M O D U L E  *-*-*-*-*-*-*-*-*-*-

!  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
!  Principal author: Nick Gould
!  History -
!   originally released GALAHAD Version 2.1. April 25th 2004

!  For full documentation, see 
!   http://galahad.rl.ac.uk/galahad-www/specs.html

  MODULE GALAHAD_DEMO_double

    USE GALAHAD_SYMBOLS
    USE GALAHAD_SPECFILE_double

    IMPLICIT NONE     

    PRIVATE
    PUBLIC :: DEMO_initialize, DEMO_read_specfile, DEMO_main, DEMO_terminate

!---------------------
!   P r e c i s i o n
!---------------------

    INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )

!--------------------------
!  Derived type definitions
!--------------------------

    TYPE, PUBLIC :: DEMO_control_type
      INTEGER :: error, out, print_level
      LOGICAL :: space_critical, deallocate_error_fatal
      CHARACTER ( LEN = 30 ) :: prefix
    END TYPE DEMO_control_type

    TYPE, PUBLIC :: DEMO_inform_type
      INTEGER :: status, alloc_status
      CHARACTER ( LEN = 80 ) :: bad_alloc
    END TYPE DEMO_inform_type

    TYPE, PUBLIC :: DEMO_data_type
      REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: VECTOR
      TYPE ( DEMO_control_type ) :: control
    END TYPE DEMO_data_type

  CONTAINS

!-*-*-  G A L A H A D -  D E M O _ I N I T I A L I Z E  S U B R O U T I N E  -*-

    SUBROUTINE DEMO_initialize( data, control )

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!   Provide default values for DEMO controls

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!---------------------------------
!   D u m m y   A r g u m e n t s
!---------------------------------

    TYPE ( DEMO_data_type ), INTENT( OUT ) :: data
    TYPE ( DEMO_control_type ), INTENT( OUT ) :: control

!  Error and ordinary output unit numbers

    control%error = 6
    control%out = 6

!  Level of output required. <= 0 gives no output, = 1 gives a one-line
!  summary for every iteration, = 2 gives a summary of the inner iteration
!  for each iteration, >= 3 gives increasingly verbose (debugging) output

    control%print_level = 0

!  If space_critical is true, every effort will be made to use as little
!  space as possible. This may result in longer computation times

    control%space_critical = .FALSE.

!   If deallocate_error_fatal is true, any array/pointer deallocation error
!     will terminate execution. Otherwise, computation will continue

    control%deallocate_error_fatal  = .FALSE.

!  Each line of output from DEMO will be prefixed by %prefix

    control%prefix = '""     '

    RETURN

!  End of subroutine DEMO_initialize

    END SUBROUTINE DEMO_initialize

!-*-*-*-*-   D E M O _ R E A D _ S P E C F I L E  S U B R O U T I N E  -*-*-*-*-

    SUBROUTINE DEMO_read_specfile( control, device, alt_specname )

!  Reads the content of a specification file, and performs the assignment of 
!  values associated with given keywords to the corresponding control parameters

!  The default values as given by DEMO_initialize could (roughly) 
!  have been set as:

! BEGIN DEMO SPECIFICATIONS (DEFAULT)
!  error-printout-device                           6
!  printout-device                                 6
!  alive-device                                    40
!  print-level                                     0
!  space-critical                                  no
!  deallocate-error-fatal                          no
! END DEMO SPECIFICATIONS

!---------------------------------
!   D u m m y   A r g u m e n t s
!---------------------------------

    TYPE ( DEMO_control_type ), INTENT( INOUT ) :: control        
    INTEGER, INTENT( IN ) :: device
    CHARACTER( LEN = 16 ), OPTIONAL :: alt_specname

!---------------------------------
!   L o c a l   V a r i a b l e s
!---------------------------------

    INTEGER, PARAMETER :: lspec = 5
    CHARACTER( LEN = 16 ), PARAMETER :: specname = 'DEMO          '
    TYPE ( SPECFILE_item_type ), DIMENSION( lspec ) :: spec

!  Define the keywords

    spec%keyword = ''

!  Integer key-words

    spec( 1 )%keyword = 'error-printout-device'
    spec( 2 )%keyword = 'printout-device'
    spec( 3 )%keyword = 'print-level' 

!  Logical key-words
                                 
    spec( 4 )%keyword = 'space-critical'
    spec( 5 )%keyword = 'deallocate-error-fatal'

!  Read the specfile

    IF ( PRESENT( alt_specname ) ) THEN
      CALL SPECFILE_read( device, alt_specname, spec, lspec, control%error )
    ELSE
      CALL SPECFILE_read( device, specname, spec, lspec, control%error )
    END IF

!  Interpret the result

!  Set integer values

    CALL SPECFILE_assign_value( spec( 1 ), control%error,                     &
                                 control%error )
    CALL SPECFILE_assign_value( spec( 2 ), control%out,                       &
                                 control%error )                             
    CALL SPECFILE_assign_value( spec( 3 ), control%print_level,               &
                                 control%error )                             
!  et logical values
   
    CALL SPECFILE_assign_value( spec( 4 ), control%space_critical,            &
                                control%error )
    CALL SPECFILE_assign_value( spec( 5 ),                                    &
                                control%deallocate_error_fatal,               &
                                control%error )
    RETURN

    END SUBROUTINE DEMO_read_specfile

!-*-*-*-  G A L A H A D -  D E M O _ m a i n  S U B R O U T I N E  -*-*-*-

    SUBROUTINE DEMO_main( n, control, inform, data )

!  *-*-*-*-*-*-*-*-*-*-*-*-  A R G U M E N T S  -*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
!
!  For full details see the specification sheet for GALAHAD_DEMO. 
!
! n is a scalar variable of type default integer, that might hold the
!  dimension of a problem - this is an example!
!
! control is a scalar variable of type DEMO_control_type. See DEMO_initialize
!  for details
!
! inform is a scalar variable of type DEMO_inform_type. On initial entry, 
!  inform%status should be set to 1. On exit, the following components will have 
!  been set:
!
!  status is a scalar variable of type default integer, that gives
!   the exit status from the package. Possible values are:
!
!     0. The run was succesful
!
!    -1. An allocation error occurred. A message indicating the offending
!        array is written on unit control%error, and the returned allocation 
!        status and a string containing the name of the offending array
!        are held in inform%alloc_status and inform%bad_alloc respectively.
!    -2. A deallocation error occurred.  A message indicating the offending 
!        array is written on unit control%error and the returned allocation 
!        status and a string containing the name of the offending array
!        are held in inform%alloc_status and inform%bad_alloc respectively.
!    -3. The restriction %n > 0 has been violated.
!
!  alloc_status is a scalar variable of type default integer, that gives
!   the status of the last attempted array allocation or deallocation.
!   This will be 0 if status = 0.
!
!  bad_alloc is a scalar variable of type default character
!   and length 80, that  gives the name of the last internal array 
!   for which there were allocation or deallocation errors.
!   This will be the null string if status = 0. 
!
!  data is a scalar variable of type DEMO_data_type used for internal data.

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!---------------------------------
!   D u m m y   A r g u m e n t s
!---------------------------------

     INTEGER, INTENT( IN ) :: n
     TYPE ( DEMO_control_type ), INTENT( IN ) :: control
     TYPE ( DEMO_inform_type ), INTENT( INOUT ) :: inform
     TYPE ( DEMO_data_type ), INTENT( INOUT ) :: data

!---------------------------------
!   L o c a l   V a r i a b l e s
!---------------------------------

     CHARACTER ( LEN = 80 ) :: array_name
 
!  main parts of the code

!  set initial values

    inform%status = 0 ; inform%alloc_status = 0 ; inform%bad_alloc = 0

!  ensure that input parameters are within allowed ranges

     IF ( n <= 0 ) THEN
       inform%status = GALAHAD_error_restrictions
       RETURN
     END IF

!  allocate sufficient space for the problem

     array_name = 'demo: data%VECTOR'
     CALL SPACE_resize_array( n, data%X_current, inform%status,                &
            inform%alloc_status, array_name = array_name,                      &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
     IF ( inform%status /= 0 ) RETURN

!  other executable statements follow ....

    RETURN

!  End of subroutine DEMO_main

    END SUBROUTINE DEMO_main

!-*-*-  G A L A H A D -  D E M O _ t e r m i n a t e  S U B R O U T I N E  -*-*-

    SUBROUTINE DEMO_terminate( data, control, inform )

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!   Deallocate all private storage

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!---------------------------------
!   D u m m y   A r g u m e n t s
!---------------------------------

    TYPE ( DEMO_data_type ), INTENT( INOUT ) :: data
    TYPE ( DEMO_control_type ), INTENT( IN ) :: control
    TYPE ( DEMO_inform_type ), INTENT( INOUT ) :: inform
 
!---------------------------------
!   L o c a l   V a r i a b l e s
!---------------------------------

   CHARACTER ( LEN = 80 ) :: array_name

!  Deallocate all remaining allocated arrays

    array_name = 'demo: data%VECTOR'
    CALL SPACE_dealloc_array( data%VECTOR,                                     &
       inform%status, inform%alloc_status, array_name = array_name,            &
       bad_alloc = inform%bad_alloc, out = control%error )
    IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

    RETURN

!  End of subroutine DEMO_terminate

    END SUBROUTINE DEMO_terminate

!  End of module GALAHAD_DEMO

  END MODULE GALAHAD_DEMO_double











