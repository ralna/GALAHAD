! THIS VERSION: GALAHAD 5.3 - 2025-08-27 AT 16:40 GMT

#include "ssids_routines.h"
#include "ssids_procedures.h"

!  COPYRIGHT (c) 2016 The Science and Technology Facilities Council (STFC)
!  authors: Jonathan Hogg and Florent Lopez
!  licence: BSD licence, see LICENCE file for details
!  Forked and extended for GALAHAD, Nick Gould, version 3.1, 2016

  MODULE GALAHAD_SSIDS_profile
    USE GALAHAD_KINDS
    IMPLICIT NONE

    PRIVATE
    PUBLIC :: profile_begin, profile_end, profile_task_type,                   &
              profile_create_task, profile_set_state, profile_add_event

    TYPE :: profile_task_type
      PRIVATE
      TYPE(C_PTR) :: ctask
    CONTAINS
      PROCEDURE :: end_task
    END TYPE profile_task_type

#ifdef INTEGER_64
    INTERFACE
      SUBROUTINE c_begin( nregions, regions )                                  &
            BIND( C, name = "galahad_ssids_profile_begin_64" )
        USE, INTRINSIC :: iso_c_binding
        IMPLICIT NONE
        INTEGER( c_int ), value :: nregions
        TYPE( c_ptr ), value, INTENT( in ) :: regions
      END SUBROUTINE c_begin

      SUBROUTINE profile_end(  )                                               &
            BIND( C, name = "galahad_ssids_profile_end_64" )
      END SUBROUTINE profile_end

      TYPE( C_PTR ) FUNCTION c_create_task( name, thread )                     &
            BIND( C, name = "galahad_ssids_profile_create_task_64" )
         USE, INTRINSIC :: iso_c_binding
         CHARACTER( C_CHAR ), DIMENSION( * ), INTENT( IN ) :: name
         INTEGER( C_INT ), value :: thread
      END FUNCTION c_create_task

      SUBROUTINE c_end_task( task )                                            &
            BIND( C, name = "galahad_ssids_profile_end_task_64" )
         USE, INTRINSIC :: iso_c_binding
         type( C_PTR ), value :: task
      END SUBROUTINE c_end_task

      SUBROUTINE c_set_state( container, type, name )                          &
            BIND( C, name = "galahad_ssids_profile_set_state_64" )
         USE, INTRINSIC :: iso_c_binding
         CHARACTER( C_CHAR ), DIMENSION( * ), INTENT( IN ) :: container
         CHARACTER( C_CHAR ), DIMENSION( * ), INTENT( IN ) :: type
         CHARACTER( C_CHAR ), DIMENSION( * ), INTENT( IN ) :: name
      END SUBROUTINE c_set_state

      SUBROUTINE c_add_event( type, val, thread )                              &
        BIND( C, name = "galahad_ssids_profile_add_event_64" )
        USE, INTRINSIC :: iso_c_binding
        IMPLICIT NONE
        CHARACTER( C_CHAR ), DIMENSION( * ), INTENT( IN ) :: type
        CHARACTER( C_CHAR ), DIMENSION( * ), INTENT( IN ) :: val
        INTEGER( C_INT ), value :: thread
      END SUBROUTINE c_add_event
    END INTERFACE
#else
    INTERFACE
      SUBROUTINE c_begin( nregions, regions )                                  &
            BIND( C, name = "galahad_ssids_profile_begin" )
        USE, INTRINSIC :: iso_c_binding
        IMPLICIT NONE
        INTEGER( c_int ), value :: nregions
        TYPE( c_ptr ), value, INTENT( IN ) :: regions
      END SUBROUTINE c_begin

      SUBROUTINE profile_end(  )                                               &
            BIND( C, name = "galahad_ssids_profile_end" )
      END SUBROUTINE profile_end

      TYPE( C_PTR ) FUNCTION c_create_task( name, thread )                     &
            BIND( C, name = "galahad_ssids_profile_create_task" )
         USE, INTRINSIC :: iso_c_binding
         CHARACTER( C_CHAR ), DIMENSION( * ), INTENT( IN ) :: name
         INTEGER( C_INT ), value :: thread
      END FUNCTION c_create_task

      SUBROUTINE c_end_task( task )                                            &
            BIND( C, name = "galahad_ssids_profile_end_task" )
         USE, INTRINSIC :: iso_c_binding
         type( C_PTR ), value :: task
      END SUBROUTINE c_end_task

      SUBROUTINE c_set_state( container, type, name )                          &
            BIND( C, name = "galahad_ssids_profile_set_state" )
         USE, INTRINSIC :: iso_c_binding
         CHARACTER( C_CHAR ), DIMENSION( * ), INTENT( IN ) :: container
         CHARACTER( C_CHAR ), DIMENSION( * ), INTENT( IN ) :: type
         CHARACTER( C_CHAR ), DIMENSION( * ), INTENT( IN ) :: name
      END SUBROUTINE c_set_state

      SUBROUTINE c_add_event( type, val, thread ) &
        BIND( C, name = "galahad_ssids_profile_add_event" )
        USE, INTRINSIC :: iso_c_binding
        IMPLICIT NONE
        CHARACTER( C_CHAR ), DIMENSION( * ), INTENT( IN ) :: type
        CHARACTER( C_CHAR ), DIMENSION( * ), INTENT( IN ) :: val
        INTEGER( C_INT ), value :: thread
      END SUBROUTINE c_add_event
   END INTERFACE
#endif

  CONTAINS

!-  G A L A H A D -  S S I D S _   S U B R O U T I N E -

    SUBROUTINE profile_begin( regions )
    USE GALAHAD_HW, ONLY : HW_numa_region, HW_c_numa_region
    IMPLICIT NONE

    TYPE( HW_numa_region ), DIMENSION( : ), INTENT( IN ) :: regions

    TYPE( HW_c_numa_region ), DIMENSION( : ), POINTER, CONTIGUOUS :: f_regions
    INTEGER( c_int ) :: nregions
    INTEGER( ip_ ) :: ngpus
    INTEGER( ip_ ) :: i
    INTEGER( ip_ ) :: st
    INTEGER( c_int ), DIMENSION( : ), POINTER, CONTIGUOUS :: gpus
    TYPE( c_ptr ) :: c_regions

    NULLIFY( gpus )

    nregions = SIZE( regions, 1 )
    ALLOCATE( f_regions( nregions ), stat=st )
    DO i = 1, nregions
      f_regions( i )%nproc = INT( regions( i )%nproc, C_INT )
      ngpus = SIZE( regions( i )%gpus, 1 )
      f_regions( i )%ngpu = INT( ngpus, C_INT )
      IF ( ngpus .gt. 0 ) THEN
        ALLOCATE( gpus( ngpus ), stat=st )
        gpus( : ) = INT( regions( i )%gpus,kind = c_int )
        f_regions( i )%gpus = C_LOC( gpus( 1 ) )
        NULLIFY( gpus )
      END IF
    END DO

    c_regions = C_LOC( f_regions )

    CALL c_begin( nregions, c_regions )

    ! TODO free data structures

    END SUBROUTINE profile_begin

!-  G A L A H A D -  S S I D S _ profile_create_task  F U N C T I O N  -

    TYPE( profile_task_type ) FUNCTION profile_create_task( name, thread )
    CHARACTER( LEN = * ), INTENT( IN ) :: name
    INTEGER( ip_ ), OPTIONAL, INTENT( IN ) :: thread

    INTEGER( C_INT ) :: mythread
    CHARACTER( C_CHAR ), DIMENSION( 200 ) :: cname

    mythread = -1 ! autodetect
    IF ( PRESENT( thread ) ) mythread = INT( thread,kind = C_INT )
    CALL f2c_string( name, cname )

    profile_create_task%ctask = c_create_task( cname, mythread )
    END FUNCTION profile_create_task

    SUBROUTINE end_task( this )
    CLASS( profile_task_type ), INTENT( IN ) :: this

    CALL c_end_task( this%ctask )
    END SUBROUTINE end_task

!-  G A L A H A D -  S S I D S _ p r o f i l e _ set_state  S U B R O U T I N E 

    SUBROUTINE profile_set_state( container, type, name )
    CHARACTER( LEN = * ), INTENT( IN ) :: container
    CHARACTER( LEN = * ), INTENT( IN ) :: type
    CHARACTER( LEN = * ), INTENT( IN ) :: name

    CHARACTER( C_CHAR ), DIMENSION( 200 ) :: cname, ctype, ccontainer

    CALL f2c_string( container, ccontainer )
    CALL f2c_string( type, ctype )
    CALL f2c_string( name, cname )
    CALL c_set_state( ccontainer, ctype, cname )
    RETURN

    END SUBROUTINE profile_set_state

!-  G A L A H A D -  S S I D S _ p r o f i l e _ add_event  S U B R O U T I N E 

    SUBROUTINE profile_add_event( type, val, thread )
    IMPLICIT NONE

    CHARACTER( LEN = * ), INTENT( IN ) :: type
    CHARACTER( LEN = * ), INTENT( IN ) :: val
    INTEGER( ip_ ), OPTIONAL, INTENT( IN ) :: thread

    INTEGER( C_INT ) :: mythread
    CHARACTER( C_CHAR ), DIMENSION( 200 ) :: ctype, cval

    CALL f2c_string( type, ctype )
    CALL f2c_string( val, cval )
    mythread = -1 ! autodetect
    IF ( PRESENT( thread ) ) mythread = INT( thread,kind = C_INT )

    CALL c_add_event( ctype, cval, mythread )
    RETURN

    END SUBROUTINE profile_add_event

!-  G A L A H A D -  S S I D S _ f 2 c _ s t r i n g  S U B R O U T I N E -

    SUBROUTINE f2c_string( fstring, cstring, stat )

!  convert Fortran CHARACTER to C string, adding null terminator.
!   fstring Fortran string to convert.
!   cstring On output, overwritten with C string. Must be long enough
!     to include null termination.
!   stat Status, 0 on sucess, otherwise number of additional CHARACTERs
!     required.

    CHARACTER( LEN = * ), INTENT( IN ) :: fstring
    CHARACTER( C_CHAR ), DIMENSION( : ), INTENT( OUT ) :: cstring
    INTEGER( ip_ ), OPTIONAL, INTENT( OUT ) :: stat

    INTEGER( ip_ ) :: i

    IF (  SIZE( cstring ) < LEN( fstring ) + 1 ) THEN

!  not big enough, need +1 for null terminator

      IF ( PRESENT( stat ) ) stat = LEN( fstring ) + 1 - SIZE( cstring )
      RETURN
    END IF

    DO i = 1, LEN( fstring )
      cstring( i ) = fstring( i:i )
    END do
    cstring( LEN( fstring ) + 1 ) = C_NULL_CHAR
    RETURN

    END SUBROUTINE f2c_string

  END MODULE GALAHAD_SSIDS_profile
