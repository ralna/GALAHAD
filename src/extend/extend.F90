! THIS VERSION: GALAHAD 4.1 - 2023-01-24 AT 09:30 GMT.

#include "galahad_modules.h"

!-*-*-*-*-*-*-*-*-*-  G A L A H A D _ E X T E N D   M O D U L E  -*-*-*-*-*-*-*-

!  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
!  Principal author: Nick Gould

!  History -
!   originally released as part of LANCELOT, August 23rd 1995
!   rebadged for GALAHAD Version 3.3. May 20th 2021

!  For full documentation, see
!   http://galahad.rl.ac.uk/galahad-www/specs.html

!  Nick Gould, for GALAHAD productions
!  Copyright reserved
!

  MODULE GALAHAD_EXTEND_precision

        USE GALAHAD_KINDS_precision

        IMPLICIT NONE

        PRIVATE
        PUBLIC :: EXTEND_save_type, EXTEND_arrays

!  Define generic interfaces to routines for extending allocatable arrays

        INTERFACE EXTEND_arrays
           MODULE PROCEDURE EXTEND_array_real, EXTEND_array_integer
        END INTERFACE

!  =================================
!  The EXTEND_save_type derived type
!  =================================

        TYPE :: EXTEND_save_type
         INTEGER ( KIND = ip_ ) :: lirnh, ljcnh, llink_min, lirnh_min
         INTEGER ( KIND = ip_ ) :: ljcnh_min, lh_min, lh, litran_min
         INTEGER ( KIND = ip_ ) :: lwtran_min, lwtran, litran, l_link_e_u_v
         INTEGER ( KIND = ip_ ) :: llink, lrowst, lpos, lused, lfilled
        END TYPE EXTEND_save_type

      CONTAINS

!  Module procedures

         SUBROUTINE EXTEND_array_real( ARRAY, old_length, used_length,         &
             new_length, min_length, buffer, status, alloc_status )

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

         INTEGER ( KIND = ip_ ), INTENT( IN ) :: old_length, buffer
         INTEGER ( KIND = ip_ ), INTENT( OUT ) :: status, alloc_status
         INTEGER ( KIND = ip_ ), INTENT( INOUT ) :: used_length, min_length
         INTEGER ( KIND = ip_ ), INTENT( INOUT ) :: new_length
         REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : ) :: ARRAY

!  local variables

         INTEGER ( KIND = ip_ ) :: length
         LOGICAL :: file_open
         REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : ) :: DUMMY

!  Make sure that the new length is larger than the old

         IF ( new_length <= old_length ) new_length = 2 * old_length

!  Ensure that the input data is consistent

         used_length = MIN( used_length, old_length )
         min_length = MAX( old_length + 1, MIN( min_length, new_length ) )

!  If possible, allocate DUMMY to hold the old values of ARRAY

         ALLOCATE( DUMMY( used_length ), STAT = alloc_status )

!  If the allocation failed, resort to using an external unit

         IF ( alloc_status /= 0 ) GO TO 100

         DUMMY( : used_length ) = ARRAY( : used_length )

!  Extend the length of ARRAY

         DEALLOCATE( ARRAY )
         length = new_length

  10     CONTINUE
         ALLOCATE( ARRAY( length ), STAT = alloc_status )

!  If the allocation failed, reduce the new length and retry

         IF ( alloc_status /= 0 ) THEN
            length = length + ( length - min_length ) / 2

!  If there is insufficient room for both ARRAY and DUMMY, use an external unit

            IF ( length < min_length ) THEN

!  Rewind the buffer i/o unit

               INQUIRE( UNIT = buffer, OPENED = file_open )
               IF ( file_open ) THEN
                  REWIND( UNIT = buffer )
               ELSE
                  OPEN( UNIT = buffer )
               END IF

!  Copy the contents of ARRAY into the buffer i/o area

               WRITE( UNIT = buffer, FMT = * ) DUMMY( : used_length )

!  Extend the length of ARRAY

               DEALLOCATE( DUMMY )
               GO TO 110
            END IF
            GO TO 10
         END IF

!  Copy the contents of ARRAY back from the buffer i/o area

         ARRAY( : used_length ) = DUMMY( : used_length )
         DEALLOCATE( DUMMY )
         new_length = length
         GO TO 200

!  Use an external unit for writing

 100     CONTINUE

!  Rewind the buffer i/o unit

         INQUIRE( UNIT = buffer, OPENED = file_open )
         IF ( file_open ) THEN
            REWIND( UNIT = buffer )
         ELSE
            OPEN( UNIT = buffer )
         END IF

!  Copy the contents of ARRAY into the buffer i/o area

         WRITE( UNIT = buffer, FMT = * ) ARRAY( : used_length )

!  Extend the length of ARRAY

         DEALLOCATE( ARRAY )

 110     CONTINUE
         ALLOCATE( ARRAY( new_length ), STAT = alloc_status )

!  If the allocation failed, reduce the new length and retry

         IF ( alloc_status /= 0 ) THEN
            new_length = min_length + ( new_length - min_length ) / 2
            IF ( new_length < min_length ) THEN
               status = 12
               RETURN
            END IF
            GO TO 110
         END IF

!  Copy the contents of ARRAY back from the buffer i/o area

         REWIND( UNIT = buffer )
         READ( UNIT = buffer, FMT = * ) ARRAY( : used_length )

!  Successful exit

   200   CONTINUE
         status = 0
         RETURN

!  End of subroutine EXTEND_array_real

         END SUBROUTINE EXTEND_array_real

         SUBROUTINE EXTEND_array_integer( ARRAY, old_length, used_length,      &
             new_length, min_length, buffer, status, alloc_status )

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

         INTEGER ( KIND = ip_ ), INTENT( IN ) :: old_length, buffer
         INTEGER ( KIND = ip_ ), INTENT( OUT ) :: status, alloc_status
         INTEGER ( KIND = ip_ ), INTENT( INOUT ) :: used_length, min_length
         INTEGER ( KIND = ip_ ), INTENT( INOUT ) :: new_length
         INTEGER ( KIND = ip_ ), ALLOCATABLE, DIMENSION( : ) :: ARRAY

         INTEGER ( KIND = ip_ ) :: length
         LOGICAL :: file_open
         INTEGER ( KIND = ip_ ), ALLOCATABLE, DIMENSION( : ) :: DUMMY

!  Make sure that the new length is larger than the old

         IF ( new_length <= old_length ) new_length = 2 * old_length

!  Ensure that the input data is consistent

         used_length = MIN( used_length, old_length )
         min_length = MAX( old_length + 1, MIN( min_length, new_length ) )

!  If possible, allocate DUMMY to hold the old values of ARRAY

         ALLOCATE( DUMMY( used_length ), STAT = alloc_status )

!  If the allocation failed, resort to using an external unit

         IF ( alloc_status /= 0 ) GO TO 100

         DUMMY( : used_length ) = ARRAY( : used_length )

!  Extend the length of ARRAY

         DEALLOCATE( ARRAY )
         length = new_length

  10     CONTINUE
         ALLOCATE( ARRAY( length ), STAT = alloc_status )

!  If the allocation failed, reduce the new length and retry

         IF ( alloc_status /= 0 ) THEN
            length = length + ( length - min_length ) / 2

!  If there is insufficient room for both ARRAY and DUMMY, use an external unit

            IF ( length < min_length ) THEN

!  Rewind the buffer i/o unit

               INQUIRE( UNIT = buffer, OPENED = file_open )
               IF ( file_open ) THEN
                  REWIND( UNIT = buffer )
               ELSE
                  OPEN( UNIT = buffer )
               END IF

!  Copy the contents of ARRAY into the buffer i/o area

               WRITE( UNIT = buffer, FMT = * ) DUMMY( : used_length )

!  Extend the length of ARRAY

               DEALLOCATE( DUMMY )
               GO TO 110
            END IF
            GO TO 10
         END IF

!  Copy the contents of ARRAY back from the buffer i/o area

         ARRAY( : used_length ) = DUMMY( : used_length )
         DEALLOCATE( DUMMY )
         new_length = length
         GO TO 200

!  Use an external unit for writing

 100     CONTINUE

!  Rewind the buffer i/o unit

         INQUIRE( UNIT = buffer, OPENED = file_open )
         IF ( file_open ) THEN
            REWIND( UNIT = buffer )
         ELSE
            OPEN( UNIT = buffer )
         END IF

!  Copy the contents of ARRAY into the buffer i/o area

         WRITE( UNIT = buffer, FMT = * ) ARRAY( : used_length )

!  Extend the length of ARRAY

         DEALLOCATE( ARRAY )

 110     CONTINUE
         ALLOCATE( ARRAY( new_length ), STAT = alloc_status )

!  If the allocation failed, reduce the new length and retry

         IF ( alloc_status /= 0 ) THEN
            new_length = min_length + ( new_length - min_length ) / 2
            IF ( new_length < min_length ) THEN
               status = 12
               RETURN
            END IF
            GO TO 110
         END IF

!  Copy the contents of ARRAY back from the buffer i/o area

         REWIND( UNIT = buffer )
         READ( UNIT = buffer, FMT = * ) ARRAY( : used_length )

!  Successful exit

   200   CONTINUE
         status = 0
         RETURN

!  End of subroutine EXTEND_array_integer

         END SUBROUTINE EXTEND_array_integer

!  End of module GALAHAD_EXTEND

  END MODULE GALAHAD_EXTEND_precision

