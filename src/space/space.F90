! THIS VERSION: GALAHAD 4.3 - 2024-01-26 AT 10:40 GMT.

#include "galahad_modules.h"

!-*-*-*-*-*-*-*-  G A L A H A D _ S P A C E   M O D U L E  *-*-*-*-*-*-*-*-*-*

!  This module contains simple routines for possibly changing the size
!  of allocatable or pointer arrays, and for deallocating them after use

!  Nick Gould, for GALAHAD productions
!  Copyright reserved
!  January 27th 2005

   MODULE GALAHAD_SPACE_precision

     USE GALAHAD_SYMBOLS
     USE GALAHAD_KINDS_precision
     USE GALAHAD_SMT_precision

     IMPLICIT NONE

     PRIVATE
     PUBLIC :: SPACE_resize_pointer, SPACE_resize_array,                       &
               SPACE_resize_cpointer, SPACE_resize_carray,                     &
               SPACE_extend_array, SPACE_extend_carray,                        &
               SPACE_dealloc_pointer, SPACE_dealloc_array,                     &
               SPACE_dealloc_smt_type

!  generic interfaces to cope with both pointer and allocatable
!  real and integer arrays

     INTERFACE SPACE_resize_pointer
       MODULE PROCEDURE SPACE_resize_real_pointer,                             &
                        SPACE_resize_reallu_pointer,                           &
                        SPACE_resize_real2_pointer,                            &
                        SPACE_resize_integer_pointer,                          &
                        SPACE_resize_integerlu_pointer,                        &
                        SPACE_resize_integer2_pointer,                         &
                        SPACE_resize_logical_pointer,                          &
                        SPACE_resize_logical64_pointer,                        &
                        SPACE_resize_character_pointer,                        &
                        SPACE_resize_character2_pointer
     END INTERFACE SPACE_resize_pointer

     INTERFACE SPACE_resize_cpointer
       MODULE PROCEDURE SPACE_resize_real_cpointer,                            &
                        SPACE_resize_integer_cpointer,                         &
                        SPACE_resize_logical_cpointer
     END INTERFACE SPACE_resize_cpointer

     INTERFACE SPACE_resize_array
       MODULE PROCEDURE SPACE_resize_real_array,                               &
                        SPACE_resize_reallu_array,                             &
                        SPACE_resize_real2_array,                              &
                        SPACE_resize_reallu2_array,                            &
                        SPACE_resize_reallulu_array,                           &
                        SPACE_resize_reallu3_array,                            &
                        SPACE_resize_reallulu3_array,                          &
                        SPACE_resize_complex_array,                            &
                        SPACE_resize_integer_array,                            &
                        SPACE_resize_integerlu_array,                          &
                        SPACE_resize_integer2_array,                           &
                        SPACE_resize_logical_array,                            &
                        SPACE_resize_logical64_array,                          &
                        SPACE_resize_character_array,                          &
                        SPACE_resize_character2_array
     END INTERFACE SPACE_resize_array

     INTERFACE SPACE_resize_carray
       MODULE PROCEDURE SPACE_resize_real_carray,                              &
                        SPACE_resize_integer_carray,                           &
                        SPACE_resize_logical_array
     END INTERFACE SPACE_resize_carray

     INTERFACE SPACE_extend_array
       MODULE PROCEDURE SPACE_extend_array_integer,                            &
                        SPACE_extend_array_real,                               &
                        SPACE_extend_array_logical
     END INTERFACE SPACE_extend_array

     INTERFACE SPACE_extend_carray
       MODULE PROCEDURE SPACE_extend_carray_integer,                           &
                        SPACE_extend_carray_real
     END INTERFACE SPACE_extend_carray

     INTERFACE SPACE_dealloc_pointer
       MODULE PROCEDURE SPACE_dealloc_real_pointer,                            &
                        SPACE_dealloc_real2_pointer,                           &
                        SPACE_dealloc_integer_pointer,                         &
                        SPACE_dealloc_integer2_pointer,                        &
                        SPACE_dealloc_logical_pointer,                         &
                        SPACE_dealloc_logical64_pointer,                       &
                        SPACE_dealloc_character_pointer,                       &
                        SPACE_dealloc_character2_pointer
     END INTERFACE SPACE_dealloc_pointer

     INTERFACE SPACE_dealloc_array
       MODULE PROCEDURE SPACE_dealloc_real_array,                              &
                        SPACE_dealloc_real2_array,                             &
                        SPACE_dealloc_real3_array,                             &
                        SPACE_dealloc_complex_array,                           &
                        SPACE_dealloc_integer_array,                           &
                        SPACE_dealloc_integer2_array,                          &
                        SPACE_dealloc_logical_array,                           &
                        SPACE_dealloc_logical64_array,                         &
                        SPACE_dealloc_character_array,                         &
                        SPACE_dealloc_character2_array
     END INTERFACE SPACE_dealloc_array

   CONTAINS

!  -  S P A C E _ R E S I Z E _ R E A L _ P O I N T E R  S U B R O U T I N E -

     SUBROUTINE SPACE_resize_real_pointer( len, point, status, alloc_status,   &
       deallocate_error_fatal, point_name, exact_size, bad_alloc, out )

!  Ensure that the real pointer array "point" is of length at least len.

!  If exact_size is prsent and true, point is reallocated to be of size len.
!  Otherwise point is only reallocated if its length is currently smaller
!  than len

!  Dummy arguments

     INTEGER ( KIND = ip_ ), INTENT( IN ) :: len
     INTEGER ( KIND = ip_ ), INTENT( OUT ) :: status, alloc_status
     REAL ( KIND = rp_ ), POINTER, DIMENSION( : ) :: point
     INTEGER ( KIND = ip_ ), OPTIONAL :: out
     LOGICAL, OPTIONAL :: deallocate_error_fatal, exact_size
     CHARACTER ( LEN = 80 ), OPTIONAL :: point_name
     CHARACTER ( LEN = 80 ), OPTIONAL :: bad_alloc

!  Local variable

     LOGICAL :: reallocate

!  Check to see if a reallocation (or initial allocation) is needed

     status = GALAHAD_ok ; alloc_status = 0 ; reallocate = .TRUE.
     IF ( PRESENT( bad_alloc ) ) bad_alloc = ''
     IF ( ASSOCIATED( point ) ) THEN
       IF ( PRESENT( exact_size ) ) THEN
         IF ( exact_size ) THEN
           IF ( SIZE( point ) /= len ) THEN
             CALL SPACE_dealloc_pointer( point, status, alloc_status,          &
                                         point_name, bad_alloc, out )
           ELSE ; reallocate = .FALSE.
           END IF
         ELSE
           IF ( SIZE( point ) < len ) THEN
             CALL SPACE_dealloc_pointer( point, status, alloc_status,          &
                                         point_name, bad_alloc, out )
           ELSE ; reallocate = .FALSE.
           END IF
         END IF
       ELSE
         IF ( SIZE( point ) < len ) THEN
           CALL SPACE_dealloc_pointer( point, status, alloc_status,            &
                                       point_name, bad_alloc, out )
          ELSE ; reallocate = .FALSE.
          END IF
       END IF
     END IF

!  If a deallocation error occured, return if desired

     IF ( PRESENT( deallocate_error_fatal ) ) THEN
       IF ( deallocate_error_fatal .AND. alloc_status /= 0 ) THEN
         status = GALAHAD_error_deallocate ; RETURN
       END IF
     ELSE
       IF ( alloc_status /= 0 ) THEN
         status = GALAHAD_error_deallocate ; RETURN
       END IF
     END IF

!  Reallocate point to be of length len, checking for error returns

     IF ( reallocate ) ALLOCATE( point( len ), STAT = alloc_status )
     IF ( alloc_status /= 0 ) THEN
       status = GALAHAD_error_allocate
       IF ( PRESENT( bad_alloc ) .AND. PRESENT( point_name ) )                 &
         bad_alloc = point_name
       IF ( PRESENT( out ) ) THEN
         IF ( PRESENT( point_name ) ) THEN
           IF ( out > 0 ) WRITE( out, 2900 ) TRIM( point_name ), alloc_status
         ELSE
           IF ( out > 0 ) WRITE( out, 2910 ) alloc_status
         END IF
       END IF
     END IF
     RETURN

!  Non-executable statements

2900 FORMAT( ' ** Allocation error for ', A, /, '     status = ', I6 )
2910 FORMAT( ' ** Allocation error status = ', I6 )

!  End of SPACE_resize_real_pointer

     END SUBROUTINE SPACE_resize_real_pointer

! -  S P A C E _ R E S I Z E _ R E A L L U _ P O I N T E R  S U B R O U T I N E

     SUBROUTINE SPACE_resize_reallu_pointer( l, u, point, status, alloc_status,&
       deallocate_error_fatal, point_name, exact_size, bad_alloc, out )

!  Ensure that the real pointer array "point" has bounds at least l and u

!  If exact_size is prsent and true, point is reallocated to have bounds l and u
!  Otherwise point is only reallocated if its bounds are insufficient to cover
!  l and u

!  Dummy arguments

     INTEGER ( KIND = ip_ ), INTENT( IN ) :: l, u
     INTEGER ( KIND = ip_ ), INTENT( OUT ) :: status, alloc_status
     REAL ( KIND = rp_ ), POINTER, DIMENSION( : ) :: point
     INTEGER ( KIND = ip_ ), OPTIONAL :: out
     LOGICAL, OPTIONAL :: deallocate_error_fatal, exact_size
     CHARACTER ( LEN = 80 ), OPTIONAL :: point_name
     CHARACTER ( LEN = 80 ), OPTIONAL :: bad_alloc

!  Local variable

     LOGICAL :: reallocate

!  Check to see if a reallocation (or initial allocation) is needed

     status = GALAHAD_ok ; alloc_status = 0 ; reallocate = .TRUE.
     IF ( PRESENT( bad_alloc ) ) bad_alloc = ''
     IF ( ASSOCIATED( point ) ) THEN
       IF ( PRESENT( exact_size ) ) THEN
         IF ( exact_size ) THEN
           IF ( LBOUND( point, 1 ) /= l .OR. UBOUND( point, 1 ) /= u ) THEN
             CALL SPACE_dealloc_pointer( point, status, alloc_status,          &
                                         point_name, bad_alloc, out )
           ELSE ; reallocate = .FALSE.
           END IF
         ELSE
           IF ( LBOUND( point, 1 ) > l .OR. UBOUND( point, 1 ) < u ) THEN
             CALL SPACE_dealloc_pointer( point, status, alloc_status,          &
                                         point_name, bad_alloc, out )
           ELSE ; reallocate = .FALSE.
           END IF
         END IF
       ELSE
         IF ( LBOUND( point, 1 ) > l .OR. UBOUND( point, 1 ) < u ) THEN
           CALL SPACE_dealloc_pointer( point, status, alloc_status,            &
                                       point_name, bad_alloc, out )
          ELSE ; reallocate = .FALSE.
          END IF
       END IF
     END IF

!  If a deallocation error occured, return if desired

     IF ( PRESENT( deallocate_error_fatal ) ) THEN
       IF ( deallocate_error_fatal .AND. alloc_status /= 0 ) THEN
         status = GALAHAD_error_deallocate ; RETURN
       END IF
     ELSE
       IF ( alloc_status /= 0 ) THEN
         status = GALAHAD_error_deallocate ; RETURN
       END IF
     END IF

!  Reallocate point to be of length len, checking for error returns

     IF ( reallocate ) THEN
       IF ( l <= u ) THEN !! avoid gfortran bug
         ALLOCATE( point( l : u ), STAT = alloc_status )
       ELSE
!        ALLOCATE( point( 0 ), STAT = alloc_status )
         alloc_status = 0
       END IF
     END IF

     IF ( alloc_status /= 0 ) THEN
       status = GALAHAD_error_allocate
       IF ( PRESENT( bad_alloc ) .AND. PRESENT( point_name ) )                 &
         bad_alloc = point_name
       IF ( PRESENT( out ) ) THEN
         IF ( PRESENT( point_name ) ) THEN
           IF ( out > 0 ) WRITE( out, 2900 ) TRIM( point_name ), alloc_status
         ELSE
           IF ( out > 0 ) WRITE( out, 2910 ) alloc_status
         END IF
       END IF
     END IF
     RETURN

!  Non-executable statements

2900 FORMAT( ' ** Allocation error for ', A, /, '     status = ', I6 )
2910 FORMAT( ' ** Allocation error status = ', I6 )

!  End of SPACE_resize_reallu_pointer

     END SUBROUTINE SPACE_resize_reallu_pointer

! -  S P A C E _ R E S I Z E _ R E A L 2 _ P O I N T E R  S U B R O U T I N E  -

     SUBROUTINE SPACE_resize_real2_pointer( len1, len2, point, status,         &
       alloc_status, deallocate_error_fatal, point_name, exact_size,           &
       bad_alloc, out )

!  Ensure that the 2D real pointer array "point" is of length at least
!  (len1,len2).

!  If exact_size is prsent and true, point is reallocated to be of size len.
!  Otherwise point is only reallocated if its length is currently smaller
!  than len

!  Dummy arguments

     INTEGER ( KIND = ip_ ), INTENT( IN ) :: len1, len2
     INTEGER ( KIND = ip_ ), INTENT( OUT ) :: status, alloc_status
     REAL ( KIND = rp_ ), POINTER, DIMENSION( : , : ) :: point
     INTEGER ( KIND = ip_ ), OPTIONAL :: out
     LOGICAL, OPTIONAL :: deallocate_error_fatal, exact_size
     CHARACTER ( LEN = 80 ), OPTIONAL :: point_name
     CHARACTER ( LEN = 80 ), OPTIONAL :: bad_alloc

!  Local variable

     LOGICAL :: reallocate

!  Check to see if a reallocation (or initial allocation) is needed

     status = GALAHAD_ok ; alloc_status = 0 ; reallocate = .TRUE.
     IF ( PRESENT( bad_alloc ) ) bad_alloc = ''
     IF ( ASSOCIATED( point ) ) THEN
       IF ( PRESENT( exact_size ) ) THEN
         IF ( exact_size ) THEN
           IF ( SIZE( point, 1 ) /= len1 .OR.                                  &
                SIZE( point, 2 ) /= len2  ) THEN
             CALL SPACE_dealloc_pointer( point, status, alloc_status,          &
                                         point_name, bad_alloc, out )
           ELSE ; reallocate = .FALSE.
           END IF
         ELSE
           IF ( SIZE( point, 1 ) /= len1 .OR.                                  &
                SIZE( point, 2 ) < len2  ) THEN
             CALL SPACE_dealloc_pointer( point, status, alloc_status,          &
                                         point_name, bad_alloc, out )
           ELSE ; reallocate = .FALSE.
           END IF
         END IF
       ELSE
         IF ( SIZE( point, 1 ) /= len1 .OR.                                    &
              SIZE( point, 2 ) < len2  ) THEN
           CALL SPACE_dealloc_pointer( point, status, alloc_status,            &
                                       point_name, bad_alloc, out )
          ELSE ; reallocate = .FALSE.
          END IF
       END IF
     END IF

!  If a deallocation error occured, return if desired

     IF ( PRESENT( deallocate_error_fatal ) ) THEN
       IF ( deallocate_error_fatal .AND. alloc_status /= 0 ) THEN
         status = GALAHAD_error_deallocate ; RETURN
       END IF
     ELSE
       IF ( alloc_status /= 0 ) THEN
         status = GALAHAD_error_deallocate ; RETURN
       END IF
     END IF

!  Reallocate point to be of length len1, len2, checking for error returns

     IF ( reallocate ) ALLOCATE( point( len1, len2 ), STAT = alloc_status )
     IF ( alloc_status /= 0 ) THEN
       status = GALAHAD_error_allocate
       IF ( PRESENT( bad_alloc ) .AND. PRESENT( point_name ) )                 &
         bad_alloc = point_name
       IF ( PRESENT( out ) ) THEN
         IF ( PRESENT( point_name ) ) THEN
           IF ( out > 0 ) WRITE( out, 2900 ) TRIM( point_name ), alloc_status
         ELSE
           IF ( out > 0 ) WRITE( out, 2910 ) alloc_status
         END IF
       END IF
     END IF
     RETURN

!  Non-executable statements

2900 FORMAT( ' ** Allocation error for ', A, /, '     status = ', I6 )
2910 FORMAT( ' ** Allocation error status = ', I6 )

!  End of SPACE_resize_real2_pointer

     END SUBROUTINE SPACE_resize_real2_pointer

! - S P A C E _ R E S I Z E _ I N T E G E R _ P O I N T E R  S U B R O U T I N E

     SUBROUTINE SPACE_resize_integer_pointer( len, point, status,              &
                   alloc_status, deallocate_error_fatal, point_name,           &
                   exact_size, bad_alloc, out )

!  Ensure that the integer pointer array "point" is of length at least len.

!  If exact_size is prsent and true, point is reallocated to be of size len.
!  Otherwise point is only reallocated if its length is currently smaller
!  than len

!  Dummy arguments

     INTEGER ( KIND = ip_ ), INTENT( IN ) :: len
     INTEGER ( KIND = ip_ ), INTENT( OUT ) :: status, alloc_status
     INTEGER ( KIND = ip_ ), POINTER, DIMENSION( : ) :: point
     INTEGER ( KIND = ip_ ), OPTIONAL :: out
     LOGICAL, OPTIONAL :: deallocate_error_fatal, exact_size
     CHARACTER ( LEN = 80 ), OPTIONAL :: point_name
     CHARACTER ( LEN = 80 ), OPTIONAL :: bad_alloc

!  Local variable

     LOGICAL :: reallocate

!  Check to see if a reallocation (or initial allocation) is needed

     status = GALAHAD_ok ; alloc_status = 0 ; reallocate = .TRUE.
     IF ( PRESENT( bad_alloc ) ) bad_alloc = ''
     IF ( ASSOCIATED( point ) ) THEN
       IF ( PRESENT( exact_size ) ) THEN
         IF ( exact_size ) THEN
           IF ( SIZE( point ) /= len ) THEN
             CALL SPACE_dealloc_pointer( point, status, alloc_status,          &
                                         point_name, bad_alloc, out )
           ELSE ; reallocate = .FALSE.
           END IF
         ELSE
           IF ( SIZE( point ) < len ) THEN
             CALL SPACE_dealloc_pointer( point, status, alloc_status,          &
                                         point_name, bad_alloc, out )
           ELSE ; reallocate = .FALSE.
           END IF
         END IF
       ELSE
         IF ( SIZE( point ) < len ) THEN
           CALL SPACE_dealloc_pointer( point, status, alloc_status,            &
                                       point_name, bad_alloc, out )
          ELSE ; reallocate = .FALSE.
          END IF
       END IF
     END IF

!  If a deallocation error occured, return if desired

     IF ( PRESENT( deallocate_error_fatal ) ) THEN
       IF ( deallocate_error_fatal .AND. alloc_status /= 0 ) THEN
         status = GALAHAD_error_deallocate ; RETURN
       END IF
     ELSE
       IF ( alloc_status /= 0 ) THEN
         status = GALAHAD_error_deallocate ; RETURN
       END IF
     END IF

!  Reallocate point to be of length len, checking for error returns

     IF ( reallocate ) ALLOCATE( point( len ), STAT = alloc_status )
     IF ( alloc_status /= 0 ) THEN
       status = GALAHAD_error_allocate
       IF ( PRESENT( bad_alloc ) .AND. PRESENT( point_name ) )                 &
         bad_alloc = point_name
       IF ( PRESENT( out ) ) THEN
         IF ( PRESENT( point_name ) ) THEN
           IF ( out > 0 ) WRITE( out, 2900 ) TRIM( point_name ), alloc_status
         ELSE
           IF ( out > 0 ) WRITE( out, 2910 ) alloc_status
         END IF
       END IF
     END IF
     RETURN

!  Non-executable statements

2900 FORMAT( ' ** Allocation error for ', A, /, '     status = ', I6 )
2910 FORMAT( ' ** Allocation error status = ', I6 )

!  End of SPACE_resize_integer_pointer

     END SUBROUTINE SPACE_resize_integer_pointer

! - S P A C E _ R E S I Z E _ I N T E G E R L U _ P O I N T E R  SUBROUTINE -

     SUBROUTINE SPACE_resize_integerlu_pointer( l, u, point, status,           &
      alloc_status, deallocate_error_fatal, point_name, exact_size,            &
      bad_alloc, out )

!  Ensure that the integer pointer array "point" has bounds at least l and u

!  If exact_size is prsent and true, point is reallocated to have bounds l and u
!  Otherwise point is only reallocated if its bounds are insufficient to cover
!  l and u

!  Dummy arguments

     INTEGER ( KIND = ip_ ), INTENT( IN ) :: l, u
     INTEGER ( KIND = ip_ ), INTENT( OUT ) :: status, alloc_status
     INTEGER ( KIND = ip_ ), POINTER, DIMENSION( : ) :: point
     INTEGER ( KIND = ip_ ), OPTIONAL :: out
     LOGICAL, OPTIONAL :: deallocate_error_fatal, exact_size
     CHARACTER ( LEN = 80 ), OPTIONAL :: point_name
     CHARACTER ( LEN = 80 ), OPTIONAL :: bad_alloc

!  Local variable

     LOGICAL :: reallocate

!  Check to see if a reallocation (or initial allocation) is needed

     status = GALAHAD_ok ; alloc_status = 0 ; reallocate = .TRUE.
     IF ( PRESENT( bad_alloc ) ) bad_alloc = ''
     IF ( ASSOCIATED( point ) ) THEN
       IF ( PRESENT( exact_size ) ) THEN
         IF ( exact_size ) THEN
           IF ( LBOUND( point, 1 ) /= l .OR. UBOUND( point, 1 ) /= u ) THEN
             CALL SPACE_dealloc_pointer( point, status, alloc_status,          &
                                         point_name, bad_alloc, out )
           ELSE ; reallocate = .FALSE.
           END IF
         ELSE
           IF ( LBOUND( point, 1 ) > l .OR. UBOUND( point, 1 ) < u ) THEN
             CALL SPACE_dealloc_pointer( point, status, alloc_status,          &
                                         point_name, bad_alloc, out )
           ELSE ; reallocate = .FALSE.
           END IF
         END IF
       ELSE
         IF ( LBOUND( point, 1 ) > l .OR. UBOUND( point, 1 ) < u ) THEN
           CALL SPACE_dealloc_pointer( point, status, alloc_status,            &
                                       point_name, bad_alloc, out )
          ELSE ; reallocate = .FALSE.
          END IF
       END IF
     END IF

!  If a deallocation error occured, return if desired

     IF ( PRESENT( deallocate_error_fatal ) ) THEN
       IF ( deallocate_error_fatal .AND. alloc_status /= 0 ) THEN
         status = GALAHAD_error_deallocate ; RETURN
       END IF
     ELSE
       IF ( alloc_status /= 0 ) THEN
         status = GALAHAD_error_deallocate ; RETURN
       END IF
     END IF

!  Reallocate point to be of length len, checking for error returns

     IF ( reallocate ) ALLOCATE( point( l : u ), STAT = alloc_status )
     IF ( alloc_status /= 0 ) THEN
       status = GALAHAD_error_allocate
       IF ( PRESENT( bad_alloc ) .AND. PRESENT( point_name ) )                 &
         bad_alloc = point_name
       IF ( PRESENT( out ) ) THEN
         IF ( PRESENT( point_name ) ) THEN
           IF ( out > 0 ) WRITE( out, 2900 ) TRIM( point_name ), alloc_status
         ELSE
           IF ( out > 0 ) WRITE( out, 2910 ) alloc_status
         END IF
       END IF
     END IF
     RETURN

!  Non-executable statements

2900 FORMAT( ' ** Allocation error for ', A, /, '     status = ', I6 )
2910 FORMAT( ' ** Allocation error status = ', I6 )

!  End of SPACE_resize_integerlu_pointer

     END SUBROUTINE SPACE_resize_integerlu_pointer

! -*-  S P A C E _ R E S I Z E _ I N T E G E R 2 _ P O I N T E R  SUBROUTINE -*-

     SUBROUTINE SPACE_resize_integer2_pointer( len1, len2, point, status,      &
       alloc_status, deallocate_error_fatal, point_name, exact_size,           &
       bad_alloc, out )

!  Ensure that the 2D integer pointer array "point" is of length at least
!  (len1,len2)

!  If exact_size is prsent and true, point is reallocated to be of size len.
!  Otherwise point is only reallocated if its length is currently smaller
!  than len

!  Dummy arguments

     INTEGER ( KIND = ip_ ), INTENT( IN ) :: len1, len2
     INTEGER ( KIND = ip_ ), INTENT( OUT ) :: status, alloc_status
     INTEGER ( KIND = ip_ ), POINTER, DIMENSION( : , : ) :: point
     INTEGER ( KIND = ip_ ), OPTIONAL :: out
     LOGICAL, OPTIONAL :: deallocate_error_fatal, exact_size
     CHARACTER ( LEN = 80 ), OPTIONAL :: point_name
     CHARACTER ( LEN = 80 ), OPTIONAL :: bad_alloc

!  Local variable

     LOGICAL :: reallocate

!  Check to see if a reallocation (or initial allocation) is needed

     status = GALAHAD_ok ; alloc_status = 0 ; reallocate = .TRUE.
     IF ( PRESENT( bad_alloc ) ) bad_alloc = ''
     IF ( ASSOCIATED( point ) ) THEN
       IF ( PRESENT( exact_size ) ) THEN
         IF ( exact_size ) THEN
           IF ( SIZE( point, 1 ) /= len1 .OR.                                  &
                SIZE( point, 2 ) /= len2  ) THEN
             CALL SPACE_dealloc_pointer( point, status, alloc_status,          &
                                         point_name, bad_alloc, out )
           ELSE ; reallocate = .FALSE.
           END IF
         ELSE
           IF ( SIZE( point, 1 ) /= len1 .OR.                                  &
                SIZE( point, 2 ) < len2  ) THEN
             CALL SPACE_dealloc_pointer( point, status, alloc_status,          &
                                         point_name, bad_alloc, out )
           ELSE ; reallocate = .FALSE.
           END IF
         END IF
       ELSE
         IF ( SIZE( point, 1 ) /= len1 .OR.                                    &
              SIZE( point, 2 ) < len2  ) THEN
           CALL SPACE_dealloc_pointer( point, status, alloc_status,            &
                                       point_name, bad_alloc, out )
          ELSE ; reallocate = .FALSE.
          END IF
       END IF
     END IF

!  If a deallocation error occured, return if desired

     IF ( PRESENT( deallocate_error_fatal ) ) THEN
       IF ( deallocate_error_fatal .AND. alloc_status /= 0 ) THEN
         status = GALAHAD_error_deallocate ; RETURN
       END IF
     ELSE
       IF ( alloc_status /= 0 ) THEN
         status = GALAHAD_error_deallocate ; RETURN
       END IF
     END IF

!  Reallocate point to be of length len1, len2, checking for error returns

     IF ( reallocate ) ALLOCATE( point( len1, len2 ), STAT = alloc_status )
     IF ( alloc_status /= 0 ) THEN
       status = GALAHAD_error_allocate
       IF ( PRESENT( bad_alloc ) .AND. PRESENT( point_name ) )                 &
         bad_alloc = point_name
       IF ( PRESENT( out ) ) THEN
         IF ( PRESENT( point_name ) ) THEN
           IF ( out > 0 ) WRITE( out, 2900 ) TRIM( point_name ), alloc_status
         ELSE
           IF ( out > 0 ) WRITE( out, 2910 ) alloc_status
         END IF
       END IF
     END IF
     RETURN

!  Non-executable statements

2900 FORMAT( ' ** Allocation error for ', A, /, '     status = ', I6 )
2910 FORMAT( ' ** Allocation error status = ', I6 )

!  End of SPACE_resize_integer2_pointer

     END SUBROUTINE SPACE_resize_integer2_pointer

! - S P A C E _ R E S I Z E _ L O G I C A L _ P O I N T E R  S U B R O U T I N E

     SUBROUTINE SPACE_resize_logical_pointer( len, point, status, alloc_status,&
       deallocate_error_fatal, point_name, exact_size, bad_alloc, out )

!  Ensure that the logical pointer array "point" is of length at least len.

!  If exact_size is prsent and true, point is reallocated to be of size len.
!  Otherwise point is only reallocated if its length is currently smaller
!  than len

!  Dummy arguments

     INTEGER ( KIND = ip_ ), INTENT( IN ) :: len
     INTEGER ( KIND = ip_ ), INTENT( OUT ) :: status, alloc_status
     LOGICAL ( KIND = i4_ ), POINTER, DIMENSION( : ) :: point
     INTEGER ( KIND = ip_ ), OPTIONAL :: out
     LOGICAL, OPTIONAL :: deallocate_error_fatal, exact_size
     CHARACTER ( LEN = 80 ), OPTIONAL :: point_name
     CHARACTER ( LEN = 80 ), OPTIONAL :: bad_alloc

!  Local variable

     LOGICAL :: reallocate

!  Check to see if a reallocation (or initial allocation) is needed

     status = GALAHAD_ok ; alloc_status = 0 ; reallocate = .TRUE.
     IF ( PRESENT( bad_alloc ) ) bad_alloc = ''
     IF ( ASSOCIATED( point ) ) THEN
       IF ( PRESENT( exact_size ) ) THEN
         IF ( exact_size ) THEN
           IF ( SIZE( point ) /= len ) THEN
             CALL SPACE_dealloc_pointer( point, status, alloc_status,          &
                                         point_name, bad_alloc, out )
           ELSE ; reallocate = .FALSE.
           END IF
         ELSE
           IF ( SIZE( point ) < len ) THEN
             CALL SPACE_dealloc_pointer( point, status, alloc_status,          &
                                         point_name, bad_alloc, out )
           ELSE ; reallocate = .FALSE.
           END IF
         END IF
       ELSE
         IF ( SIZE( point ) < len ) THEN
           CALL SPACE_dealloc_pointer( point, status, alloc_status,            &
                                       point_name, bad_alloc, out )
          ELSE ; reallocate = .FALSE.
          END IF
       END IF
     END IF

!  If a deallocation error occured, return if desired

     IF ( PRESENT( deallocate_error_fatal ) ) THEN
       IF ( deallocate_error_fatal .AND. alloc_status /= 0 ) THEN
         status = GALAHAD_error_deallocate ; RETURN
       END IF
     ELSE
       IF ( alloc_status /= 0 ) THEN
         status = GALAHAD_error_deallocate ; RETURN
       END IF
     END IF

!  Reallocate point to be of length len, checking for error returns

     IF ( reallocate ) ALLOCATE( point( len ), STAT = alloc_status )
     IF ( alloc_status /= 0 ) THEN
       status = GALAHAD_error_allocate
       IF ( PRESENT( bad_alloc ) .AND. PRESENT( point_name ) )                 &
         bad_alloc = point_name
       IF ( PRESENT( out ) ) THEN
         IF ( PRESENT( point_name ) ) THEN
           IF ( out > 0 ) WRITE( out, 2900 ) TRIM( point_name ), alloc_status
         ELSE
           IF ( out > 0 ) WRITE( out, 2910 ) alloc_status
         END IF
       END IF
     END IF
     RETURN

!  Non-executable statements

2900 FORMAT( ' ** Allocation error for ', A, /, '     status = ', I6 )
2910 FORMAT( ' ** Allocation error status = ', I6 )

!  End of SPACE_resize_logical_pointer

     END SUBROUTINE SPACE_resize_logical_pointer

! S P A C E _ R E S I Z E _ L O G I C A L 6 4 _ P O I N T E R  S U B R O U T INE

     SUBROUTINE SPACE_resize_logical64_pointer( len, point, status,            &
                                       alloc_status, deallocate_error_fatal,   &
                                       point_name, exact_size, bad_alloc, out )

!  Ensure that the logical pointer array "point" is of length at least len.

!  If exact_size is prsent and true, point is reallocated to be of size len.
!  Otherwise point is only reallocated if its length is currently smaller
!  than len

!  Dummy arguments

     INTEGER ( KIND = ip_ ), INTENT( IN ) :: len
     INTEGER ( KIND = ip_ ), INTENT( OUT ) :: status, alloc_status
     LOGICAL ( KIND = i8_ ), POINTER, DIMENSION( : ) :: point
     INTEGER ( KIND = ip_ ), OPTIONAL :: out
     LOGICAL, OPTIONAL :: deallocate_error_fatal, exact_size
     CHARACTER ( LEN = 80 ), OPTIONAL :: point_name
     CHARACTER ( LEN = 80 ), OPTIONAL :: bad_alloc

!  Local variable

     LOGICAL :: reallocate

!  Check to see if a reallocation (or initial allocation) is needed

     status = GALAHAD_ok ; alloc_status = 0 ; reallocate = .TRUE.
     IF ( PRESENT( bad_alloc ) ) bad_alloc = ''
     IF ( ASSOCIATED( point ) ) THEN
       IF ( PRESENT( exact_size ) ) THEN
         IF ( exact_size ) THEN
           IF ( SIZE( point ) /= len ) THEN
             CALL SPACE_dealloc_pointer( point, status, alloc_status,          &
                                         point_name, bad_alloc, out )
           ELSE ; reallocate = .FALSE.
           END IF
         ELSE
           IF ( SIZE( point ) < len ) THEN
             CALL SPACE_dealloc_pointer( point, status, alloc_status,          &
                                         point_name, bad_alloc, out )
           ELSE ; reallocate = .FALSE.
           END IF
         END IF
       ELSE
         IF ( SIZE( point ) < len ) THEN
           CALL SPACE_dealloc_pointer( point, status, alloc_status,            &
                                       point_name, bad_alloc, out )
          ELSE ; reallocate = .FALSE.
          END IF
       END IF
     END IF

!  If a deallocation error occured, return if desired

     IF ( PRESENT( deallocate_error_fatal ) ) THEN
       IF ( deallocate_error_fatal .AND. alloc_status /= 0 ) THEN
         status = GALAHAD_error_deallocate ; RETURN
       END IF
     ELSE
       IF ( alloc_status /= 0 ) THEN
         status = GALAHAD_error_deallocate ; RETURN
       END IF
     END IF

!  Reallocate point to be of length len, checking for error returns

     IF ( reallocate ) ALLOCATE( point( len ), STAT = alloc_status )
     IF ( alloc_status /= 0 ) THEN
       status = GALAHAD_error_allocate
       IF ( PRESENT( bad_alloc ) .AND. PRESENT( point_name ) )                 &
         bad_alloc = point_name
       IF ( PRESENT( out ) ) THEN
         IF ( PRESENT( point_name ) ) THEN
           IF ( out > 0 ) WRITE( out, 2900 ) TRIM( point_name ), alloc_status
         ELSE
           IF ( out > 0 ) WRITE( out, 2910 ) alloc_status
         END IF
       END IF
     END IF
     RETURN

!  Non-executable statements

2900 FORMAT( ' ** Allocation error for ', A, /, '     status = ', I6 )
2910 FORMAT( ' ** Allocation error status = ', I6 )

!  End of SPACE_resize_logical64_pointer

     END SUBROUTINE SPACE_resize_logical64_pointer

! -*- S P A C E _ R E S I Z E _ C H A R A C T E R _ P O I N T E R  SUBROUTINE -

     SUBROUTINE SPACE_resize_character_pointer( len, point, status,            &
       alloc_status, deallocate_error_fatal, point_name, exact_size,           &
       bad_alloc, out )

!  Ensure that the character pointer array "point" is of length at least len.

!  If exact_size is prsent and true, point is reallocated to be of size len.
!  Otherwise point is only reallocated if its length is currently smaller
!  than len

!  Dummy arguments

     INTEGER ( KIND = ip_ ), INTENT( IN ) :: len
     INTEGER ( KIND = ip_ ), INTENT( OUT ) :: status, alloc_status
     CHARACTER( LEN = * ), POINTER, DIMENSION( : ) :: point
     INTEGER ( KIND = ip_ ), OPTIONAL :: out
     LOGICAL, OPTIONAL :: deallocate_error_fatal, exact_size
     CHARACTER ( LEN = 80 ), OPTIONAL :: point_name
     CHARACTER ( LEN = 80 ), OPTIONAL :: bad_alloc

!  Local variable

     LOGICAL :: reallocate

!  Check to see if a reallocation (or initial allocation) is needed

     status = GALAHAD_ok ; alloc_status = 0 ; reallocate = .TRUE.
     IF ( PRESENT( bad_alloc ) ) bad_alloc = ''
     IF ( ASSOCIATED( point ) ) THEN
       IF ( PRESENT( exact_size ) ) THEN
         IF ( exact_size ) THEN
           IF ( SIZE( point ) /= len ) THEN
             CALL SPACE_dealloc_pointer( point, status, alloc_status,          &
                                         point_name, bad_alloc, out )
           ELSE ; reallocate = .FALSE.
           END IF
         ELSE
           IF ( SIZE( point ) < len ) THEN
             CALL SPACE_dealloc_pointer( point, status, alloc_status,          &
                                         point_name, bad_alloc, out )
           ELSE ; reallocate = .FALSE.
           END IF
         END IF
       ELSE
         IF ( SIZE( point ) < len ) THEN
           CALL SPACE_dealloc_pointer( point, status, alloc_status,            &
                                       point_name, bad_alloc, out )
          ELSE ; reallocate = .FALSE.
          END IF
       END IF
     END IF

!  If a deallocation error occured, return if desired

     IF ( PRESENT( deallocate_error_fatal ) ) THEN
       IF ( deallocate_error_fatal .AND. alloc_status /= 0 ) THEN
         status = GALAHAD_error_deallocate ; RETURN
       END IF
     ELSE
       IF ( alloc_status /= 0 ) THEN
         status = GALAHAD_error_deallocate ; RETURN
       END IF
     END IF

!  Reallocate point to be of length len, checking for error returns

     IF ( reallocate ) ALLOCATE( point( len ), STAT = alloc_status )
     IF ( alloc_status /= 0 ) THEN
       status = GALAHAD_error_allocate
       IF ( PRESENT( bad_alloc ) .AND. PRESENT( point_name ) )                 &
         bad_alloc = point_name
       IF ( PRESENT( out ) ) THEN
         IF ( PRESENT( point_name ) ) THEN
           IF ( out > 0 ) WRITE( out, 2900 ) TRIM( point_name ), alloc_status
         ELSE
           IF ( out > 0 ) WRITE( out, 2910 ) alloc_status
         END IF
       END IF
     END IF
     RETURN

!  Non-executable statements

2900 FORMAT( ' ** Allocation error for ', A, /, '     status = ', I6 )
2910 FORMAT( ' ** Allocation error status = ', I6 )

!  End of SPACE_resize_character_pointer

     END SUBROUTINE SPACE_resize_character_pointer

! -  S P A C E _ R E S I Z E _ C H A R A C T E R 2 _ P O I N T E R SUBROUTINE  -

     SUBROUTINE SPACE_resize_character2_pointer( len1, len2, point, status,    &
       alloc_status, deallocate_error_fatal, point_name, exact_size,           &
       bad_alloc, out )

!  Ensure that the character pointer array "point" is of length at least
!  (len1,len2)

!  If exact_size is prsent and true, point is reallocated to be of size len.
!  Otherwise point is only reallocated if its length is currently smaller
!  than len

!  Dummy arguments

     INTEGER ( KIND = ip_ ), INTENT( IN ) :: len1, len2
     INTEGER ( KIND = ip_ ), INTENT( OUT ) :: status, alloc_status
     CHARACTER( LEN = * ), POINTER, DIMENSION( : , : ) :: point
     INTEGER ( KIND = ip_ ), OPTIONAL :: out
     LOGICAL, OPTIONAL :: deallocate_error_fatal, exact_size
     CHARACTER ( LEN = 80 ), OPTIONAL :: point_name
     CHARACTER ( LEN = 80 ), OPTIONAL :: bad_alloc

!  Local variable

     LOGICAL :: reallocate

!  Check to see if a reallocation (or initial allocation) is needed

     status = GALAHAD_ok ; alloc_status = 0 ; reallocate = .TRUE.
     IF ( PRESENT( bad_alloc ) ) bad_alloc = ''
     IF ( ASSOCIATED( point ) ) THEN
       IF ( PRESENT( exact_size ) ) THEN
         IF ( exact_size ) THEN
           IF ( SIZE( point, 1 ) /= len1 .OR.                                  &
                SIZE( point, 2 ) /= len2  ) THEN
             CALL SPACE_dealloc_pointer( point, status, alloc_status,          &
                                         point_name, bad_alloc, out )
           ELSE ; reallocate = .FALSE.
           END IF
         ELSE
           IF ( SIZE( point, 1 ) /= len1 .OR.                                  &
                SIZE( point, 2 ) < len2  ) THEN
             CALL SPACE_dealloc_pointer( point, status, alloc_status,          &
                                         point_name, bad_alloc, out )
           ELSE ; reallocate = .FALSE.
           END IF
         END IF
       ELSE
         IF ( SIZE( point, 1 ) /= len1 .OR.                                    &
              SIZE( point, 2 ) < len2  ) THEN
           CALL SPACE_dealloc_pointer( point, status, alloc_status,            &
                                       point_name, bad_alloc, out )
          ELSE ; reallocate = .FALSE.
          END IF
       END IF
     END IF

!  If a deallocation error occured, return if desired

     IF ( PRESENT( deallocate_error_fatal ) ) THEN
       IF ( deallocate_error_fatal .AND. alloc_status /= 0 ) THEN
         status = GALAHAD_error_deallocate ; RETURN
       END IF
     ELSE
       IF ( alloc_status /= 0 ) THEN
         status = GALAHAD_error_deallocate ; RETURN
       END IF
     END IF

!  Reallocate point to be of length len1, len2, checking for error returns

     IF ( reallocate ) ALLOCATE( point( len1, len2 ), STAT = alloc_status )
     IF ( alloc_status /= 0 ) THEN
       status = GALAHAD_error_allocate
       IF ( PRESENT( bad_alloc ) .AND. PRESENT( point_name ) )                 &
         bad_alloc = point_name
       IF ( PRESENT( out ) ) THEN
         IF ( PRESENT( point_name ) ) THEN
           IF ( out > 0 ) WRITE( out, 2900 ) TRIM( point_name ), alloc_status
         ELSE
           IF ( out > 0 ) WRITE( out, 2910 ) alloc_status
         END IF
       END IF
     END IF
     RETURN

!  Non-executable statements

2900 FORMAT( ' ** Allocation error for ', A, /, '     status = ', I6 )
2910 FORMAT( ' ** Allocation error status = ', I6 )

!  End of SPACE_resize_character2_pointer

     END SUBROUTINE SPACE_resize_character2_pointer

!  - S P A C E _ R E S I Z E _ R E A L _ C P O I N T E R  S U B R O U T I N E -

     SUBROUTINE SPACE_resize_real_cpointer( len, point, status, alloc_status,  &
       deallocate_error_fatal, point_name, exact_size, bad_alloc, out )

!  Ensure that the real c-style pointer array "point" is of length at least len.

!  If exact_size is prsent and true, point is reallocated to be of size len.
!  Otherwise point is only reallocated if its length is currently smaller
!  than len

!  Dummy arguments

     INTEGER ( KIND = ip_ ), INTENT( IN ) :: len
     INTEGER ( KIND = ip_ ), INTENT( OUT ) :: status, alloc_status
     REAL ( KIND = rp_ ), POINTER, DIMENSION( : ) :: point
     INTEGER ( KIND = ip_ ), OPTIONAL :: out
     LOGICAL, OPTIONAL :: deallocate_error_fatal, exact_size
     CHARACTER ( LEN = 80 ), OPTIONAL :: point_name
     CHARACTER ( LEN = 80 ), OPTIONAL :: bad_alloc

!  Local variable

     LOGICAL :: reallocate

!  Check to see if a reallocation (or initial allocation) is needed

     status = GALAHAD_ok ; alloc_status = 0 ; reallocate = .TRUE.
     IF ( PRESENT( bad_alloc ) ) bad_alloc = ''
     IF ( ASSOCIATED( point ) ) THEN
       IF ( PRESENT( exact_size ) ) THEN
         IF ( exact_size ) THEN
           IF ( LBOUND( point, 1 ) /= 0 .OR.                                   &
                UBOUND( point, 1 ) /= len - 1 ) THEN
             CALL SPACE_dealloc_pointer( point, status, alloc_status,          &
                                         point_name, bad_alloc, out )
           ELSE ; reallocate = .FALSE.
           END IF
         ELSE
           IF ( LBOUND( point, 1 ) /= 0 .OR.                                   &
                UBOUND( point, 1 ) < len - 1 ) THEN
             CALL SPACE_dealloc_pointer( point, status, alloc_status,          &
                                         point_name, bad_alloc, out )
           ELSE ; reallocate = .FALSE.
           END IF
         END IF
       ELSE
           IF ( LBOUND( point, 1 ) /= 0 .OR.                                   &
                UBOUND( point, 1 ) < len - 1 ) THEN
           CALL SPACE_dealloc_pointer( point, status, alloc_status,            &
                                       point_name, bad_alloc, out )
          ELSE ; reallocate = .FALSE.
          END IF
       END IF
     END IF

!  If a deallocation error occured, return if desired

     IF ( PRESENT( deallocate_error_fatal ) ) THEN
       IF ( deallocate_error_fatal .AND. alloc_status /= 0 ) THEN
         status = GALAHAD_error_deallocate ; RETURN
       END IF
     ELSE
       IF ( alloc_status /= 0 ) THEN
         status = GALAHAD_error_deallocate ; RETURN
       END IF
     END IF

!  Reallocate point to be of length len, checking for error returns

     IF ( reallocate ) ALLOCATE( point( 0 : len - 1 ), STAT = alloc_status )
     IF ( alloc_status /= 0 ) THEN
       status = GALAHAD_error_allocate
       IF ( PRESENT( bad_alloc ) .AND. PRESENT( point_name ) )                 &
         bad_alloc = point_name
       IF ( PRESENT( out ) ) THEN
         IF ( PRESENT( point_name ) ) THEN
           IF ( out > 0 ) WRITE( out, 2900 ) TRIM( point_name ), alloc_status
         ELSE
           IF ( out > 0 ) WRITE( out, 2910 ) alloc_status
         END IF
       END IF
     END IF
     RETURN

!  Non-executable statements

2900 FORMAT( ' ** Allocation error for ', A, /, '     status = ', I6 )
2910 FORMAT( ' ** Allocation error status = ', I6 )

!  End of SPACE_resize_real_cpointer

     END SUBROUTINE SPACE_resize_real_cpointer

!  - S P A C E _ R E S I Z E _ I N T E G E R  _ C P O I N T E R  SUBROUTINE -

     SUBROUTINE SPACE_resize_integer_cpointer( len, point, status,             &
       alloc_status, deallocate_error_fatal, point_name, exact_size,           &
       bad_alloc, out )

!  Ensure that the integer c-style pointer array "point" is of length at least
!  len.

!  If exact_size is prsent and true, point is reallocated to be of size len.
!  Otherwise point is only reallocated if its length is currently smaller
!  than len

!  Dummy arguments

     INTEGER ( KIND = ip_ ), INTENT( IN ) :: len
     INTEGER ( KIND = ip_ ), INTENT( OUT ) :: status, alloc_status
     INTEGER ( KIND = ip_ ), POINTER, DIMENSION( : ) :: point
     INTEGER ( KIND = ip_ ), OPTIONAL :: out
     LOGICAL, OPTIONAL :: deallocate_error_fatal, exact_size
     CHARACTER ( LEN = 80 ), OPTIONAL :: point_name
     CHARACTER ( LEN = 80 ), OPTIONAL :: bad_alloc

!  Local variable

     LOGICAL :: reallocate

!  Check to see if a reallocation (or initial allocation) is needed

     status = GALAHAD_ok ; alloc_status = 0 ; reallocate = .TRUE.
     IF ( PRESENT( bad_alloc ) ) bad_alloc = ''
     IF ( ASSOCIATED( point ) ) THEN
       IF ( PRESENT( exact_size ) ) THEN
         IF ( exact_size ) THEN
           IF ( LBOUND( point, 1 ) /= 0 .OR.                                   &
                UBOUND( point, 1 ) /= len - 1 ) THEN
             CALL SPACE_dealloc_pointer( point, status, alloc_status,          &
                                         point_name, bad_alloc, out )
           ELSE ; reallocate = .FALSE.
           END IF
         ELSE
           IF ( LBOUND( point, 1 ) /= 0 .OR.                                   &
                UBOUND( point, 1 ) < len - 1 ) THEN
             CALL SPACE_dealloc_pointer( point, status, alloc_status,          &
                                         point_name, bad_alloc, out )
           ELSE ; reallocate = .FALSE.
           END IF
         END IF
       ELSE
           IF ( LBOUND( point, 1 ) /= 0 .OR.                                   &
                UBOUND( point, 1 ) < len - 1 ) THEN
           CALL SPACE_dealloc_pointer( point, status, alloc_status,            &
                                       point_name, bad_alloc, out )
          ELSE ; reallocate = .FALSE.
          END IF
       END IF
     END IF

!  If a deallocation error occured, return if desired

     IF ( PRESENT( deallocate_error_fatal ) ) THEN
       IF ( deallocate_error_fatal .AND. alloc_status /= 0 ) THEN
         status = GALAHAD_error_deallocate ; RETURN
       END IF
     ELSE
       IF ( alloc_status /= 0 ) THEN
         status = GALAHAD_error_deallocate ; RETURN
       END IF
     END IF

!  Reallocate point to be of length len, checking for error returns

     IF ( reallocate ) ALLOCATE( point( 0 : len - 1 ), STAT = alloc_status )
     IF ( alloc_status /= 0 ) THEN
       status = GALAHAD_error_allocate
       IF ( PRESENT( bad_alloc ) .AND. PRESENT( point_name ) )                 &
         bad_alloc = point_name
       IF ( PRESENT( out ) ) THEN
         IF ( PRESENT( point_name ) ) THEN
           IF ( out > 0 ) WRITE( out, 2900 ) TRIM( point_name ), alloc_status
         ELSE
           IF ( out > 0 ) WRITE( out, 2910 ) alloc_status
         END IF
       END IF
     END IF
     RETURN

!  Non-executable statements

2900 FORMAT( ' ** Allocation error for ', A, /, '     status = ', I6 )
2910 FORMAT( ' ** Allocation error status = ', I6 )

!  End of SPACE_resize_integer_cpointer

     END SUBROUTINE SPACE_resize_integer_cpointer

!  - S P A C E _ R E S I Z E _ L O G I C A L  _ C P O I N T E R  SUBROUTINE -

     SUBROUTINE SPACE_resize_logical_cpointer( len, point, status,             &
       alloc_status, deallocate_error_fatal, point_name, exact_size,           &
       bad_alloc, out )

!  Ensure that the logical c-style pointer array "point" is of length at least
!  len.

!  If exact_size is prsent and true, point is reallocated to be of size len.
!  Otherwise point is only reallocated if its length is currently smaller
!  than len

!  Dummy arguments

     INTEGER ( KIND = ip_ ), INTENT( IN ) :: len
     INTEGER ( KIND = ip_ ), INTENT( OUT ) :: status, alloc_status
     LOGICAL ( KIND = lp_ ), POINTER, DIMENSION( : ) :: point
     INTEGER ( KIND = ip_ ), OPTIONAL :: out
     LOGICAL, OPTIONAL :: deallocate_error_fatal, exact_size
     CHARACTER ( LEN = 80 ), OPTIONAL :: point_name
     CHARACTER ( LEN = 80 ), OPTIONAL :: bad_alloc

!  Local variable

     LOGICAL :: reallocate

!  Check to see if a reallocation (or initial allocation) is needed

     status = GALAHAD_ok ; alloc_status = 0 ; reallocate = .TRUE.
     IF ( PRESENT( bad_alloc ) ) bad_alloc = ''
     IF ( ASSOCIATED( point ) ) THEN
       IF ( PRESENT( exact_size ) ) THEN
         IF ( exact_size ) THEN
           IF ( LBOUND( point, 1 ) /= 0 .OR.                                   &
                UBOUND( point, 1 ) /= len - 1 ) THEN
             CALL SPACE_dealloc_pointer( point, status, alloc_status,          &
                                         point_name, bad_alloc, out )
           ELSE ; reallocate = .FALSE.
           END IF
         ELSE
           IF ( LBOUND( point, 1 ) /= 0 .OR.                                   &
                UBOUND( point, 1 ) < len - 1 ) THEN
             CALL SPACE_dealloc_pointer( point, status, alloc_status,          &
                                         point_name, bad_alloc, out )
           ELSE ; reallocate = .FALSE.
           END IF
         END IF
       ELSE
           IF ( LBOUND( point, 1 ) /= 0 .OR.                                   &
                UBOUND( point, 1 ) < len - 1 ) THEN
           CALL SPACE_dealloc_pointer( point, status, alloc_status,            &
                                       point_name, bad_alloc, out )
          ELSE ; reallocate = .FALSE.
          END IF
       END IF
     END IF

!  If a deallocation error occured, return if desired

     IF ( PRESENT( deallocate_error_fatal ) ) THEN
       IF ( deallocate_error_fatal .AND. alloc_status /= 0 ) THEN
         status = GALAHAD_error_deallocate ; RETURN
       END IF
     ELSE
       IF ( alloc_status /= 0 ) THEN
         status = GALAHAD_error_deallocate ; RETURN
       END IF
     END IF

!  Reallocate point to be of length len, checking for error returns

     IF ( reallocate ) ALLOCATE( point( 0 : len - 1 ), STAT = alloc_status )
     IF ( alloc_status /= 0 ) THEN
       status = GALAHAD_error_allocate
       IF ( PRESENT( bad_alloc ) .AND. PRESENT( point_name ) )                 &
         bad_alloc = point_name
       IF ( PRESENT( out ) ) THEN
         IF ( PRESENT( point_name ) ) THEN
           IF ( out > 0 ) WRITE( out, 2900 ) TRIM( point_name ), alloc_status
         ELSE
           IF ( out > 0 ) WRITE( out, 2910 ) alloc_status
         END IF
       END IF
     END IF
     RETURN

!  Non-executable statements

2900 FORMAT( ' ** Allocation error for ', A, /, '     status = ', I6 )
2910 FORMAT( ' ** Allocation error status = ', I6 )

!  End of SPACE_resize_logical_cpointer

     END SUBROUTINE SPACE_resize_logical_cpointer

!  *-  S P A C E _ R E S I Z E _ R E A L _ A R R A Y  S U B R O U T I N E   -*-

     SUBROUTINE SPACE_resize_real_array( len, array, status, alloc_status,     &
       deallocate_error_fatal, array_name, exact_size, bad_alloc, out )

!  Ensure that the real allocatable array "array" is of length at least len.

!  If exact_size is prsent and true, array is reallocated to be of size len.
!  Otherwise array is only reallocated if its length is currently smaller
!  than len

!  Dummy arguments

     INTEGER ( KIND = ip_ ), INTENT( IN ) :: len
     INTEGER ( KIND = ip_ ), INTENT( OUT ) :: status, alloc_status
     REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : ) :: array
     INTEGER ( KIND = ip_ ), OPTIONAL :: out
     LOGICAL, OPTIONAL :: deallocate_error_fatal, exact_size
     CHARACTER ( LEN = 80 ), OPTIONAL :: array_name
     CHARACTER ( LEN = 80 ), OPTIONAL :: bad_alloc

!  Local variable

     LOGICAL :: reallocate

!  Check to see if a reallocation (or initial allocation) is needed

     status = GALAHAD_ok ; alloc_status = 0 ; reallocate = .TRUE.
     IF ( PRESENT( bad_alloc ) ) bad_alloc = ''
     IF ( ALLOCATED( array ) ) THEN
       IF ( PRESENT( exact_size ) ) THEN
         IF ( exact_size ) THEN
           IF ( SIZE( array ) /= len ) THEN
             CALL SPACE_dealloc_array( array, status, alloc_status,            &
                                       array_name, bad_alloc, out )
           ELSE ; reallocate = .FALSE.
           END IF
         ELSE
           IF ( SIZE( array ) < len ) THEN
             CALL SPACE_dealloc_array( array, status, alloc_status,            &
                                       array_name, bad_alloc, out )
           ELSE ; reallocate = .FALSE.
           END IF
         END IF
       ELSE
         IF ( SIZE( array ) < len ) THEN
           CALL SPACE_dealloc_array( array, status, alloc_status,              &
                                     array_name, bad_alloc, out )
          ELSE ; reallocate = .FALSE.
          END IF
       END IF
     END IF

!  If a deallocation error occured, return if desired

     IF ( PRESENT( deallocate_error_fatal ) ) THEN
       IF ( deallocate_error_fatal .AND. alloc_status /= 0 ) THEN
         status = GALAHAD_error_deallocate ; RETURN
       END IF
     ELSE
       IF ( alloc_status /= 0 ) THEN
         status = GALAHAD_error_deallocate ; RETURN
       END IF
     END IF

!  Reallocate array to be of length len, checking for error returns

     IF ( reallocate ) ALLOCATE( array( len ), STAT = alloc_status )
     IF ( alloc_status /= 0 ) THEN
       status = GALAHAD_error_allocate
       IF ( PRESENT( bad_alloc ) .AND. PRESENT( array_name ) )                 &
         bad_alloc = array_name
       IF ( PRESENT( out ) ) THEN
         IF ( PRESENT( array_name ) ) THEN
           IF ( out > 0 ) WRITE( out, 2900 ) TRIM( array_name ), alloc_status
         ELSE
           IF ( out > 0 ) WRITE( out, 2910 ) alloc_status
         END IF
       END IF
     END IF
     RETURN

!  Non-executable statements

2900 FORMAT( ' ** Allocation error for ', A, /, '     status = ', I6 )
2910 FORMAT( ' ** Allocation error status = ', I6 )

!  End of SPACE_resize_real_array

     END SUBROUTINE SPACE_resize_real_array

! -  S P A C E _ R E S I Z E _ R E A L L U _ A R R A Y   S U B R O U T I N E

     SUBROUTINE SPACE_resize_reallu_array( l, u, array, status, alloc_status,  &
       deallocate_error_fatal, array_name, exact_size, bad_alloc, out )

!  Ensure that the real allocatable array "array" has bounds at least l and u

!  If exact_size is prsent and true, array is reallocated to have bounds l and u
!  Otherwise array is only reallocated if its bounds are insufficient to cover
!  l and u

!  Dummy arguments

     INTEGER ( KIND = ip_ ), INTENT( IN ) :: l, u
     INTEGER ( KIND = ip_ ), INTENT( OUT ) :: status, alloc_status
     REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : ) :: array
     INTEGER ( KIND = ip_ ), OPTIONAL :: out
     LOGICAL, OPTIONAL :: deallocate_error_fatal, exact_size
     CHARACTER ( LEN = 80 ), OPTIONAL :: array_name
     CHARACTER ( LEN = 80 ), OPTIONAL :: bad_alloc

!  Local variable

     LOGICAL :: reallocate

!  Check to see if a reallocation (or initial allocation) is needed

     status = GALAHAD_ok ; alloc_status = 0 ; reallocate = .TRUE.
     IF ( PRESENT( bad_alloc ) ) bad_alloc = ''
     IF ( ALLOCATED( array ) ) THEN
       IF ( PRESENT( exact_size ) ) THEN
         IF ( exact_size ) THEN
           IF ( LBOUND( array, 1 ) /= l .OR. UBOUND( array, 1 ) /= u ) THEN
             CALL SPACE_dealloc_array( array, status, alloc_status,            &
                                       array_name, bad_alloc, out )
           ELSE ; reallocate = .FALSE.
           END IF
         ELSE
           IF ( LBOUND( array, 1 ) > l .OR. UBOUND( array, 1 ) < u ) THEN
             CALL SPACE_dealloc_array( array, status, alloc_status,            &
                                       array_name, bad_alloc, out )
           ELSE ; reallocate = .FALSE.
           END IF
         END IF
       ELSE
         IF ( LBOUND( array, 1 ) > l .OR. UBOUND( array, 1 ) < u ) THEN
           CALL SPACE_dealloc_array( array, status, alloc_status,              &
                                     array_name, bad_alloc, out )
          ELSE ; reallocate = .FALSE.
          END IF
       END IF
     END IF

!  If a deallocation error occured, return if desired

     IF ( PRESENT( deallocate_error_fatal ) ) THEN
       IF ( deallocate_error_fatal .AND. alloc_status /= 0 ) THEN
         status = GALAHAD_error_deallocate ; RETURN
       END IF
     ELSE
       IF ( alloc_status /= 0 ) THEN
         status = GALAHAD_error_deallocate ; RETURN
       END IF
     END IF

!  Reallocate array to be of length len, checking for error returns

     IF ( reallocate ) THEN
       IF ( l <= u ) THEN !! avoid gfortran bug
         ALLOCATE( array( l : u ), STAT = alloc_status )
       ELSE
         ALLOCATE( array( 0 ), STAT = alloc_status ) !comment this for gfortran?
!        alloc_status = 0
       END IF
     END IF

     IF ( alloc_status /= 0 ) THEN
       status = GALAHAD_error_allocate
       IF ( PRESENT( bad_alloc ) .AND. PRESENT( array_name ) )                 &
         bad_alloc = array_name
       IF ( PRESENT( out ) ) THEN
         IF ( PRESENT( array_name ) ) THEN
           IF ( out > 0 ) WRITE( out, 2900 ) TRIM( array_name ), alloc_status
         ELSE
           IF ( out > 0 ) WRITE( out, 2910 ) alloc_status
         END IF
       END IF
     END IF
     RETURN

!  Non-executable statements

2900 FORMAT( ' ** Allocation error for ', A, /, '     status = ', I6 )
2910 FORMAT( ' ** Allocation error status = ', I6 )

!  End of SPACE_resize_reallu_array

     END SUBROUTINE SPACE_resize_reallu_array

!  *-  S P A C E _ R E S I Z E _ R E A L 2 _ A R R A Y  S U B R O U T I N E   -*

     SUBROUTINE SPACE_resize_real2_array( len1, len2, array, status,           &
       alloc_status, deallocate_error_fatal, array_name, exact_size,           &
       bad_alloc, out )

!  Ensure that the 2D real allocatable array "array" is of length at least
!  (len1,len2)

!  If exact_size is prsent and true, array is reallocated to be of size len.
!  Otherwise array is only reallocated if its length is currently smaller
!  than len

!  Dummy arguments

     INTEGER ( KIND = ip_ ), INTENT( IN ) :: len1, len2
     INTEGER ( KIND = ip_ ), INTENT( OUT ) :: status, alloc_status
     REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : , : ) :: array
     INTEGER ( KIND = ip_ ), OPTIONAL :: out
     LOGICAL, OPTIONAL :: deallocate_error_fatal, exact_size
     CHARACTER ( LEN = 80 ), OPTIONAL :: array_name
     CHARACTER ( LEN = 80 ), OPTIONAL :: bad_alloc

!  Local variable

     LOGICAL :: reallocate

!  Check to see if a reallocation (or initial allocation) is needed

     status = GALAHAD_ok ; alloc_status = 0 ; reallocate = .TRUE.
     IF ( PRESENT( bad_alloc ) ) bad_alloc = ''
     IF ( ALLOCATED( array ) ) THEN
       IF ( PRESENT( exact_size ) ) THEN
         IF ( exact_size ) THEN
           IF ( SIZE( array, 1 ) /= len1 .OR.                                  &
                SIZE( array, 2 ) /= len2  ) THEN
             CALL SPACE_dealloc_array( array, status, alloc_status,            &
                                       array_name, bad_alloc, out )
           ELSE ; reallocate = .FALSE.
           END IF
         ELSE
           IF ( SIZE( array, 1 ) /= len1 .OR.                                  &
                SIZE( array, 2 ) < len2  ) THEN
             CALL SPACE_dealloc_array( array, status, alloc_status,            &
                                       array_name, bad_alloc, out )
           ELSE ; reallocate = .FALSE.
           END IF
         END IF
       ELSE
         IF ( SIZE( array, 1 ) /= len1 .OR.                                    &
              SIZE( array, 2 ) < len2  ) THEN
           CALL SPACE_dealloc_array( array, status, alloc_status,              &
                                       array_name, bad_alloc, out )
          ELSE ; reallocate = .FALSE.
          END IF
       END IF
     END IF

!  If a deallocation error occured, return if desired

     IF ( PRESENT( deallocate_error_fatal ) ) THEN
       IF ( deallocate_error_fatal .AND. alloc_status /= 0 ) THEN
         status = GALAHAD_error_deallocate ; RETURN
       END IF
     ELSE
       IF ( alloc_status /= 0 ) THEN
         status = GALAHAD_error_deallocate ; RETURN
       END IF
     END IF

!  Reallocate array to be of length len1, len2, checking for error returns

     IF ( reallocate ) ALLOCATE( array( len1, len2 ), STAT = alloc_status )
     IF ( alloc_status /= 0 ) THEN
       status = GALAHAD_error_allocate
       IF ( PRESENT( bad_alloc ) .AND. PRESENT( array_name ) )                 &
         bad_alloc = array_name
       IF ( PRESENT( out ) ) THEN
         IF ( PRESENT( array_name ) ) THEN
           IF ( out > 0 ) WRITE( out, 2900 ) TRIM( array_name ), alloc_status
         ELSE
           IF ( out > 0 ) WRITE( out, 2910 ) alloc_status
         END IF
       END IF
     END IF
     RETURN

!  Non-executable statements

2900 FORMAT( ' ** Allocation error for ', A, /, '     status = ', I6 )
2910 FORMAT( ' ** Allocation error status = ', I6 )

!  End of SPACE_resize_real2_array

     END SUBROUTINE SPACE_resize_real2_array

! -  S P A C E _ R E S I Z E _ R E A L L U 2 _ A R R A Y   S U B R O U T I N E

     SUBROUTINE SPACE_resize_reallu2_array( l1l, l1u, l2, array,               &
       status, alloc_status, deallocate_error_fatal,                           &
       array_name, exact_size, bad_alloc, out )

!  Ensure that the real allocatable array "array" has bounds at least l1l
!  and l1u for its first argument and l2 for its second

!  If exact_size is prsent and true, array is reallocated to have bounds l and u
!  Otherwise array is only reallocated if its bounds are insufficient to cover
!  l1l and l1u for ithe first argument and l2 for the second

!  Dummy arguments

     INTEGER ( KIND = ip_ ), INTENT( IN ) :: l1l, l1u, l2
     INTEGER ( KIND = ip_ ), INTENT( OUT ) :: status, alloc_status
     REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : , : ) :: array
     INTEGER ( KIND = ip_ ), OPTIONAL :: out
     LOGICAL, OPTIONAL :: deallocate_error_fatal, exact_size
     CHARACTER ( LEN = 80 ), OPTIONAL :: array_name
     CHARACTER ( LEN = 80 ), OPTIONAL :: bad_alloc

!  Local variable

     LOGICAL :: reallocate

!  Check to see if a reallocation (or initial allocation) is needed

     status = GALAHAD_ok ; alloc_status = 0 ; reallocate = .TRUE.
     IF ( PRESENT( bad_alloc ) ) bad_alloc = ''
     IF ( ALLOCATED( array ) ) THEN
       IF ( PRESENT( exact_size ) ) THEN
         IF ( exact_size ) THEN
           IF ( LBOUND( array, 1 ) /= l1l .OR. UBOUND( array, 1 ) /= l1u .OR.  &
                SIZE( array, 2 ) /= l2 ) THEN
             CALL SPACE_dealloc_array( array, status, alloc_status,            &
                                       array_name, bad_alloc, out )
           ELSE ; reallocate = .FALSE.
           END IF
         ELSE
           IF ( LBOUND( array, 1 ) /= l1l .OR. UBOUND( array, 1 ) /= l1u .OR.  &
                SIZE( array, 2 ) < l2 ) THEN
             CALL SPACE_dealloc_array( array, status, alloc_status,            &
                                       array_name, bad_alloc, out )
           ELSE ; reallocate = .FALSE.
           END IF
         END IF
       ELSE
         IF ( LBOUND( array, 1 ) /= l1l .OR. UBOUND( array, 1 ) /= l1u .OR.    &
              SIZE( array, 2 ) < l2 ) THEN
           CALL SPACE_dealloc_array( array, status, alloc_status,              &
                                       array_name, bad_alloc, out )
          ELSE ; reallocate = .FALSE.
          END IF
       END IF
     END IF

!  If a deallocation error occured, return if desired

     IF ( PRESENT( deallocate_error_fatal ) ) THEN
       IF ( deallocate_error_fatal .AND. alloc_status /= 0 ) THEN
         status = GALAHAD_error_deallocate ; RETURN
       END IF
     ELSE
       IF ( alloc_status /= 0 ) THEN
         status = GALAHAD_error_deallocate ; RETURN
       END IF
     END IF

!  Reallocate array to be of length len, checking for error returns

     IF ( reallocate ) ALLOCATE( array( l1l : l1u , l2 ), STAT = alloc_status )
     IF ( alloc_status /= 0 ) THEN
       status = GALAHAD_error_allocate
       IF ( PRESENT( bad_alloc ) .AND. PRESENT( array_name ) )                 &
         bad_alloc = array_name
       IF ( PRESENT( out ) ) THEN
         IF ( PRESENT( array_name ) ) THEN
           IF ( out > 0 ) WRITE( out, 2900 ) TRIM( array_name ), alloc_status
         ELSE
           IF ( out > 0 ) WRITE( out, 2910 ) alloc_status
         END IF
       END IF
     END IF
     RETURN

!  Non-executable statements

2900 FORMAT( ' ** Allocation error for ', A, /, '     status = ', I6 )
2910 FORMAT( ' ** Allocation error status = ', I6 )

!  End of SPACE_resize_reallu2_array

     END SUBROUTINE SPACE_resize_reallu2_array

!  S P A C E _ R E S I Z E _ R E A L L U L U  _ A R R A Y   S U B R O U T I N E

     SUBROUTINE SPACE_resize_reallulu_array( l1l, l1u, l2l, l2u, array,        &
       status, alloc_status, deallocate_error_fatal,                           &
       array_name, exact_size, bad_alloc, out )

!  Ensure that the real allocatable array "array" has bounds at least l1l
!  and l1u for its first argument and l21 and l2u for its second

!  If exact_size is prsent and true, array is reallocated to have bounds l and u
!  Otherwise array is only reallocated if its bounds are insufficient to cover
!  l1l and l1u for ithe first argument and l2l and l2u for the second

!  Dummy arguments

     INTEGER ( KIND = ip_ ), INTENT( IN ) :: l1l, l1u, l2l, l2u
     INTEGER ( KIND = ip_ ), INTENT( OUT ) :: status, alloc_status
     REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : , : ) :: array
     INTEGER ( KIND = ip_ ), OPTIONAL :: out
     LOGICAL, OPTIONAL :: deallocate_error_fatal, exact_size
     CHARACTER ( LEN = 80 ), OPTIONAL :: array_name
     CHARACTER ( LEN = 80 ), OPTIONAL :: bad_alloc

!  Local variable

     LOGICAL :: reallocate

!  Check to see if a reallocation (or initial allocation) is needed

     status = GALAHAD_ok ; alloc_status = 0 ; reallocate = .TRUE.
     IF ( PRESENT( bad_alloc ) ) bad_alloc = ''
     IF ( ALLOCATED( array ) ) THEN
       IF ( PRESENT( exact_size ) ) THEN
         IF ( exact_size ) THEN
           IF ( LBOUND( array, 1 ) /= l1l .OR. UBOUND( array, 1 ) /= l1u .OR.  &
                LBOUND( array, 2 ) /= l2l .OR. UBOUND( array, 2 ) /= l2u ) THEN
             CALL SPACE_dealloc_array( array, status, alloc_status,            &
                                       array_name, bad_alloc, out )
           ELSE ; reallocate = .FALSE.
           END IF
         ELSE
           IF ( LBOUND( array, 1 ) /= l1l .OR. UBOUND( array, 1 ) /= l1u .OR.  &
                LBOUND( array, 2 ) > l2l .OR. UBOUND( array, 2 ) < l2u ) THEN
             CALL SPACE_dealloc_array( array, status, alloc_status,            &
                                       array_name, bad_alloc, out )
           ELSE ; reallocate = .FALSE.
           END IF
         END IF
       ELSE
         IF ( LBOUND( array, 1 ) /= l1l .OR. UBOUND( array, 1 ) /= l1u .OR.    &
              LBOUND( array, 2 ) > l2l .OR. UBOUND( array, 2 ) < l2u ) THEN
           CALL SPACE_dealloc_array( array, status, alloc_status,              &
                                       array_name, bad_alloc, out )
          ELSE ; reallocate = .FALSE.
          END IF
       END IF
     END IF

!  If a deallocation error occured, return if desired

     IF ( PRESENT( deallocate_error_fatal ) ) THEN
       IF ( deallocate_error_fatal .AND. alloc_status /= 0 ) THEN
         status = GALAHAD_error_deallocate ; RETURN
       END IF
     ELSE
       IF ( alloc_status /= 0 ) THEN
         status = GALAHAD_error_deallocate ; RETURN
       END IF
     END IF

!  Reallocate array to be of length len, checking for error returns

     IF ( reallocate ) ALLOCATE( array( l1l : l1u , l2l : l2u ),               &
                                 STAT = alloc_status )
     IF ( alloc_status /= 0 ) THEN
       status = GALAHAD_error_allocate
       IF ( PRESENT( bad_alloc ) .AND. PRESENT( array_name ) )                 &
         bad_alloc = array_name
       IF ( PRESENT( out ) ) THEN
         IF ( PRESENT( array_name ) ) THEN
           IF ( out > 0 ) WRITE( out, 2900 ) TRIM( array_name ), alloc_status
         ELSE
           IF ( out > 0 ) WRITE( out, 2910 ) alloc_status
         END IF
       END IF
     END IF
     RETURN

!  Non-executable statements

2900 FORMAT( ' ** Allocation error for ', A, /, '     status = ', I6 )
2910 FORMAT( ' ** Allocation error status = ', I6 )

!  End of SPACE_resize_reallulu_array

     END SUBROUTINE SPACE_resize_reallulu_array

!  S P A C E _ R E S I Z E _ R E A L L U 3  _ A R R A Y   S U B R O U T I N E

     SUBROUTINE SPACE_resize_reallu3_array( l1, l2l, l2u, l3, array,           &
       status, alloc_status, deallocate_error_fatal,                           &
       array_name, exact_size, bad_alloc, out )

!  Ensure that the real allocatable array "array" has bounds at least l1
!  for its first argument, l21 and l2u for its second and l3 for its third

!  If exact_size is prsent and true, array is reallocated to have bounds l and u
!  Otherwise array is only reallocated if its bounds are insufficient to cover
!  l1 for the first argument, l2l and l2u for the second and l3 for the third

!  Dummy arguments

     INTEGER ( KIND = ip_ ), INTENT( IN ) :: l1, l2l, l2u, l3
     INTEGER ( KIND = ip_ ), INTENT( OUT ) :: status, alloc_status
     REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : , : , : ) :: array
     INTEGER ( KIND = ip_ ), OPTIONAL :: out
     LOGICAL, OPTIONAL :: deallocate_error_fatal, exact_size
     CHARACTER ( LEN = 80 ), OPTIONAL :: array_name
     CHARACTER ( LEN = 80 ), OPTIONAL :: bad_alloc

!  Local variable

     LOGICAL :: reallocate

!  Check to see if a reallocation (or initial allocation) is needed

     status = GALAHAD_ok ; alloc_status = 0 ; reallocate = .TRUE.
     IF ( PRESENT( bad_alloc ) ) bad_alloc = ''
     IF ( ALLOCATED( array ) ) THEN
       IF ( PRESENT( exact_size ) ) THEN
         IF ( exact_size ) THEN
           IF ( SIZE( array, 3 ) /= l3 .OR.                                    &
                LBOUND( array, 2 ) /= l2l .OR. UBOUND( array, 2 ) /= l2u .OR.  &
                SIZE( array, 1 ) /= l1 ) THEN
             CALL SPACE_dealloc_array( array, status, alloc_status,            &
                                       array_name, bad_alloc, out )
           ELSE ; reallocate = .FALSE.
           END IF
         ELSE
           IF ( SIZE( array, 3 ) < l3 .OR.                                     &
                LBOUND( array, 2 ) /= l2l .OR. UBOUND( array, 2 ) /= l2u .OR.  &
                SIZE( array, 1 ) /= l1 ) THEN
             CALL SPACE_dealloc_array( array, status, alloc_status,            &
                                       array_name, bad_alloc, out )
           ELSE ; reallocate = .FALSE.
           END IF
         END IF
       ELSE
         IF ( SIZE( array, 3 ) < l3 .OR.                                       &
              LBOUND( array, 2 ) /= l2l .OR. UBOUND( array, 2 ) /= l2u .OR.    &
              SIZE( array, 1 ) /= l1 ) THEN
           CALL SPACE_dealloc_array( array, status, alloc_status,              &
                                     array_name, bad_alloc, out )
          ELSE ; reallocate = .FALSE.
          END IF
       END IF
     END IF

!  If a deallocation error occured, return if desired

     IF ( PRESENT( deallocate_error_fatal ) ) THEN
       IF ( deallocate_error_fatal .AND. alloc_status /= 0 ) THEN
         status = GALAHAD_error_deallocate ; RETURN
       END IF
     ELSE
       IF ( alloc_status /= 0 ) THEN
         status = GALAHAD_error_deallocate ; RETURN
       END IF
     END IF

!  Reallocate array to be of length len, checking for error returns

     IF ( reallocate ) ALLOCATE( array( l1 , l2l : l2u , l3 ),                 &
                                 STAT = alloc_status )
     IF ( alloc_status /= 0 ) THEN
       status = GALAHAD_error_allocate
       IF ( PRESENT( bad_alloc ) .AND. PRESENT( array_name ) )                 &
         bad_alloc = array_name
       IF ( PRESENT( out ) ) THEN
         IF ( PRESENT( array_name ) ) THEN
           IF ( out > 0 ) WRITE( out, 2900 ) TRIM( array_name ), alloc_status
         ELSE
           IF ( out > 0 ) WRITE( out, 2910 ) alloc_status
         END IF
       END IF
     END IF
     RETURN

!  Non-executable statements

2900 FORMAT( ' ** Allocation error for ', A, /, '     status = ', I6 )
2910 FORMAT( ' ** Allocation error status = ', I6 )

!  End of SPACE_resize_reallu3_array

     END SUBROUTINE SPACE_resize_reallu3_array

!  S P A C E _ R E S I Z E _ R E A L L U L U 3  _ A R R A Y  S U B R O U T I N E

     SUBROUTINE SPACE_resize_reallulu3_array( l1, l2l, l2u, l3l, l3u, array,   &
       status, alloc_status, deallocate_error_fatal,                           &
       array_name, exact_size, bad_alloc, out )

!  Ensure that the real allocatable array "array" has bounds at least l1
!  for its first argument, l21 and l2u for its second and l3u and l3u for its
!  third

!  If exact_size is prsent and true, array is reallocated to have bounds l and u
!  Otherwise array is only reallocated if its bounds are insufficient to cover
!  l1 for the first argument, l2l and l2u for the second and l3l and l3u for
!  the third

!  Dummy arguments

     INTEGER ( KIND = ip_ ), INTENT( IN ) :: l1, l2l, l2u, l3l, l3u
     INTEGER ( KIND = ip_ ), INTENT( OUT ) :: status, alloc_status
     REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : , : , : ) :: array
     INTEGER ( KIND = ip_ ), OPTIONAL :: out
     LOGICAL, OPTIONAL :: deallocate_error_fatal, exact_size
     CHARACTER ( LEN = 80 ), OPTIONAL :: array_name
     CHARACTER ( LEN = 80 ), OPTIONAL :: bad_alloc

!  Local variable

     LOGICAL :: reallocate

!  Check to see if a reallocation (or initial allocation) is needed

     status = GALAHAD_ok ; alloc_status = 0 ; reallocate = .TRUE.
     IF ( PRESENT( bad_alloc ) ) bad_alloc = ''
     IF ( ALLOCATED( array ) ) THEN
       IF ( PRESENT( exact_size ) ) THEN
         IF ( exact_size ) THEN
           IF ( LBOUND( array, 3 ) /= l3l .OR. UBOUND( array, 3 ) /= l3u .OR.  &
                LBOUND( array, 2 ) /= l2l .OR. UBOUND( array, 2 ) /= l2u .OR.  &
                SIZE( array, 1 ) /= l1 ) THEN
             CALL SPACE_dealloc_array( array, status, alloc_status,            &
                                       array_name, bad_alloc, out )
           ELSE ; reallocate = .FALSE.
           END IF
         ELSE
           IF ( LBOUND( array, 3 ) > l3l .OR. UBOUND( array, 3 ) < l3u .OR.    &
                LBOUND( array, 2 ) /= l2l .OR. UBOUND( array, 2 ) /= l2u .OR.  &
                SIZE( array, 1 ) /= l1 ) THEN
             CALL SPACE_dealloc_array( array, status, alloc_status,            &
                                       array_name, bad_alloc, out )
           ELSE ; reallocate = .FALSE.
           END IF
         END IF
       ELSE
         IF ( LBOUND( array, 3 ) > l3l .OR. UBOUND( array, 3 ) < l3u .OR.      &
              LBOUND( array, 2 ) /= l2l .OR. UBOUND( array, 2 ) /= l2u .OR.    &
              SIZE( array, 1 ) /= l1 ) THEN
           CALL SPACE_dealloc_array( array, status, alloc_status,              &
                                     array_name, bad_alloc, out )
          ELSE ; reallocate = .FALSE.
          END IF
       END IF
     END IF

!  If a deallocation error occured, return if desired

     IF ( PRESENT( deallocate_error_fatal ) ) THEN
       IF ( deallocate_error_fatal .AND. alloc_status /= 0 ) THEN
         status = GALAHAD_error_deallocate ; RETURN
       END IF
     ELSE
       IF ( alloc_status /= 0 ) THEN
         status = GALAHAD_error_deallocate ; RETURN
       END IF
     END IF

!  Reallocate array to be of length len, checking for error returns

     IF ( reallocate ) ALLOCATE( array( l1 , l2l : l2u , l3l : l3u ),          &
                                 STAT = alloc_status )
     IF ( alloc_status /= 0 ) THEN
       status = GALAHAD_error_allocate
       IF ( PRESENT( bad_alloc ) .AND. PRESENT( array_name ) )                 &
         bad_alloc = array_name
       IF ( PRESENT( out ) ) THEN
         IF ( PRESENT( array_name ) ) THEN
           IF ( out > 0 ) WRITE( out, 2900 ) TRIM( array_name ), alloc_status
         ELSE
           IF ( out > 0 ) WRITE( out, 2910 ) alloc_status
         END IF
       END IF
     END IF
     RETURN

!  Non-executable statements

2900 FORMAT( ' ** Allocation error for ', A, /, '     status = ', I6 )
2910 FORMAT( ' ** Allocation error status = ', I6 )

!  End of SPACE_resize_reallulu3_array

     END SUBROUTINE SPACE_resize_reallulu3_array

! -* S P A C E _ R E S I Z E _ C O M P L E X _ A R R A Y  S U B R O U T I N E *-

     SUBROUTINE SPACE_resize_complex_array( len, array, status, alloc_status,  &
       deallocate_error_fatal, array_name, exact_size, bad_alloc, out )

!  Ensure that the complex allocatable array "array" is of length at least len.

!  If exact_size is prsent and true, array is reallocated to be of size len.
!  Otherwise array is only reallocated if its length is currently smaller
!  than len

!  Dummy arguments

     INTEGER ( KIND = ip_ ), INTENT( IN ) :: len
     INTEGER ( KIND = ip_ ), INTENT( OUT ) :: status, alloc_status
     COMPLEX ( KIND = cp_ ), ALLOCATABLE, DIMENSION( : ) :: array
     INTEGER ( KIND = ip_ ), OPTIONAL :: out
     LOGICAL, OPTIONAL :: deallocate_error_fatal, exact_size
     CHARACTER ( LEN = 80 ), OPTIONAL :: array_name
     CHARACTER ( LEN = 80 ), OPTIONAL :: bad_alloc

!  Local variable

     LOGICAL :: reallocate

!  Check to see if a reallocation (or initial allocation) is needed

     status = GALAHAD_ok ; alloc_status = 0 ; reallocate = .TRUE.
     IF ( PRESENT( bad_alloc ) ) bad_alloc = ''
     IF ( ALLOCATED( array ) ) THEN
       IF ( PRESENT( exact_size ) ) THEN
         IF ( exact_size ) THEN
           IF ( SIZE( array ) /= len ) THEN
             CALL SPACE_dealloc_array( array, status, alloc_status,            &
                                       array_name, bad_alloc, out )
           ELSE ; reallocate = .FALSE.
           END IF
         ELSE
           IF ( SIZE( array ) < len ) THEN
             CALL SPACE_dealloc_array( array, status, alloc_status,            &
                                       array_name, bad_alloc, out )
           ELSE ; reallocate = .FALSE.
           END IF
         END IF
       ELSE
         IF ( SIZE( array ) < len ) THEN
           CALL SPACE_dealloc_array( array, status, alloc_status,             &
                                     array_name, bad_alloc, out )
          ELSE ; reallocate = .FALSE.
          END IF
       END IF
     END IF

!  If a deallocation error occured, return if desired

     IF ( PRESENT( deallocate_error_fatal ) ) THEN
       IF ( deallocate_error_fatal .AND. alloc_status /= 0 ) THEN
         status = GALAHAD_error_deallocate ; RETURN
       END IF
     ELSE
       IF ( alloc_status /= 0 ) THEN
         status = GALAHAD_error_deallocate ; RETURN
       END IF
     END IF

!  Reallocate array to be of length len, checking for error returns

     IF ( reallocate ) ALLOCATE( array( len ), STAT = alloc_status )
     IF ( alloc_status /= 0 ) THEN
       status = GALAHAD_error_allocate
       IF ( PRESENT( bad_alloc ) .AND. PRESENT( array_name ) )                 &
         bad_alloc = array_name
       IF ( PRESENT( out ) ) THEN
         IF ( PRESENT( array_name ) ) THEN
           IF ( out > 0 ) WRITE( out, 2900 ) TRIM( array_name ), alloc_status
         ELSE
           IF ( out > 0 ) WRITE( out, 2910 ) alloc_status
         END IF
       END IF
     END IF
     RETURN

!  Non-executable statements

2900 FORMAT( ' ** Allocation error for ', A, /, '     status = ', I6 )
2910 FORMAT( ' ** Allocation error status = ', I6 )

!  End of SPACE_resize_complex_array

     END SUBROUTINE SPACE_resize_complex_array

!  - S P A C E _ R E S I Z E _ I N T E G E R _ A R R A Y  S U B R O U T I N E -

     SUBROUTINE SPACE_resize_integer_array( len, array, status, alloc_status,  &
       deallocate_error_fatal, array_name, exact_size, bad_alloc, out )

!  Ensure that the integer allocatable array "array" is of length at least len.

!  If exact_size is prsent and true, array is reallocated to be of size len.
!  Otherwise array is only reallocated if its length is currently smaller
!  than len

!  Dummy arguments

     INTEGER ( KIND = ip_ ), INTENT( IN ) :: len
     INTEGER ( KIND = ip_ ), INTENT( OUT ) :: status, alloc_status
     INTEGER ( KIND = ip_ ), ALLOCATABLE, DIMENSION( : ) :: array
     INTEGER ( KIND = ip_ ), OPTIONAL :: out
     LOGICAL, OPTIONAL :: deallocate_error_fatal, exact_size
     CHARACTER ( LEN = 80 ), OPTIONAL :: array_name
     CHARACTER ( LEN = 80 ), OPTIONAL :: bad_alloc

!  Local variable

     LOGICAL :: reallocate

!  Check to see if a reallocation (or initial allocation) is needed

     status = GALAHAD_ok ; alloc_status = 0 ; reallocate = .TRUE.
     IF ( PRESENT( bad_alloc ) ) bad_alloc = ''
     IF ( ALLOCATED( array ) ) THEN
       IF ( PRESENT( exact_size ) ) THEN
         IF ( exact_size ) THEN
           IF ( SIZE( array ) /= len ) THEN
             CALL SPACE_dealloc_array( array, status, alloc_status,            &
                                       array_name, bad_alloc, out )
           ELSE ; reallocate = .FALSE.
           END IF
         ELSE
           IF ( SIZE( array ) < len ) THEN
             CALL SPACE_dealloc_array( array, status, alloc_status,            &
                                       array_name, bad_alloc, out )
           ELSE ; reallocate = .FALSE.
           END IF
         END IF
       ELSE
         IF ( SIZE( array ) < len ) THEN
           CALL SPACE_dealloc_array( array, status, alloc_status,             &
                                     array_name, bad_alloc, out )
          ELSE ; reallocate = .FALSE.
          END IF
       END IF
     END IF

!  If a deallocation error occured, return if desired

     IF ( PRESENT( deallocate_error_fatal ) ) THEN
       IF ( deallocate_error_fatal .AND. alloc_status /= 0 ) THEN
         status = GALAHAD_error_deallocate ; RETURN
       END IF
     ELSE
       IF ( alloc_status /= 0 ) THEN
         status = GALAHAD_error_deallocate ; RETURN
       END IF
     END IF

!  Reallocate array to be of length len, checking for error returns

     IF ( reallocate ) ALLOCATE( array( len ), STAT = alloc_status )
     IF ( alloc_status /= 0 ) THEN
       status = GALAHAD_error_allocate
       IF ( PRESENT( bad_alloc ) .AND. PRESENT( array_name ) )                 &
         bad_alloc = array_name
       IF ( PRESENT( out ) ) THEN
         IF ( PRESENT( array_name ) ) THEN
           IF ( out > 0 ) WRITE( out, 2900 ) TRIM( array_name ), alloc_status
         ELSE
           IF ( out > 0 ) WRITE( out, 2910 ) alloc_status
         END IF
       END IF
     END IF
     RETURN

!  Non-executable statements

2900 FORMAT( ' ** Allocation error for ', A, /, '     status = ', I6 )
2910 FORMAT( ' ** Allocation error status = ', I6 )

!  End of SPACE_resize_integer_array

     END SUBROUTINE SPACE_resize_integer_array

! -*-  S P A C E _ R E S I Z E _ I N T E G E R L U _ A R R A Y   SUBROUTINE  -*-

     SUBROUTINE SPACE_resize_integerlu_array( l, u, array, status,             &
       alloc_status, deallocate_error_fatal, array_name, exact_size,           &
       bad_alloc, out )

!  Ensure that the integer allocatable array "array" has bounds at least l and u

!  If exact_size is prsent and true, array is reallocated to have bounds l and u
!  Otherwise array is only reallocated if its bounds are insufficient to cover
!  l and u

!  Dummy arguments

     INTEGER ( KIND = ip_ ), INTENT( IN ) :: l, u
     INTEGER ( KIND = ip_ ), INTENT( OUT ) :: status, alloc_status
     INTEGER ( KIND = ip_ ), ALLOCATABLE, DIMENSION( : ) :: array
     INTEGER ( KIND = ip_ ), OPTIONAL :: out
     LOGICAL, OPTIONAL :: deallocate_error_fatal, exact_size
     CHARACTER ( LEN = 80 ), OPTIONAL :: array_name
     CHARACTER ( LEN = 80 ), OPTIONAL :: bad_alloc

!  Local variable

     LOGICAL :: reallocate

!  Check to see if a reallocation (or initial allocation) is needed

     status = GALAHAD_ok ; alloc_status = 0 ; reallocate = .TRUE.
     IF ( PRESENT( bad_alloc ) ) bad_alloc = ''
     IF ( ALLOCATED( array ) ) THEN
       IF ( PRESENT( exact_size ) ) THEN
         IF ( exact_size ) THEN
           IF ( LBOUND( array, 1 ) /= l .OR. UBOUND( array, 1 ) /= u ) THEN
             CALL SPACE_dealloc_array( array, status, alloc_status,            &
                                       array_name, bad_alloc, out )
           ELSE ; reallocate = .FALSE.
           END IF
         ELSE
           IF ( LBOUND( array, 1 ) > l .OR. UBOUND( array, 1 ) < u ) THEN
             CALL SPACE_dealloc_array( array, status, alloc_status,            &
                                       array_name, bad_alloc, out )
           ELSE ; reallocate = .FALSE.
           END IF
         END IF
       ELSE
         IF ( LBOUND( array, 1 ) > l .OR. UBOUND( array, 1 ) < u ) THEN
           CALL SPACE_dealloc_array( array, status, alloc_status,              &
                                     array_name, bad_alloc, out )
          ELSE ; reallocate = .FALSE.
          END IF
       END IF
     END IF

!  If a deallocation error occured, return if desired

     IF ( PRESENT( deallocate_error_fatal ) ) THEN
       IF ( deallocate_error_fatal .AND. alloc_status /= 0 ) THEN
         status = GALAHAD_error_deallocate ; RETURN
       END IF
     ELSE
       IF ( alloc_status /= 0 ) THEN
         status = GALAHAD_error_deallocate ; RETURN
       END IF
     END IF

!  Reallocate array to be of length len, checking for error returns

     IF ( reallocate ) ALLOCATE( array( l : u ), STAT = alloc_status )
     IF ( alloc_status /= 0 ) THEN
       status = GALAHAD_error_allocate
       IF ( PRESENT( bad_alloc ) .AND. PRESENT( array_name ) )                 &
         bad_alloc = array_name
       IF ( PRESENT( out ) ) THEN
         IF ( PRESENT( array_name ) ) THEN
           IF ( out > 0 ) WRITE( out, 2900 ) TRIM( array_name ), alloc_status
         ELSE
           IF ( out > 0 ) WRITE( out, 2910 ) alloc_status
         END IF
       END IF
     END IF
     RETURN

!  Non-executable statements

2900 FORMAT( ' ** Allocation error for ', A, /, '     status = ', I6 )
2910 FORMAT( ' ** Allocation error status = ', I6 )

!  End of SPACE_resize_integerlu_array

     END SUBROUTINE SPACE_resize_integerlu_array

!  -*-  S P A C E _ R E S I Z E _ I N T E G E R 2 _ A R R A Y  SUBROUTINE   -*-

     SUBROUTINE SPACE_resize_integer2_array( len1, len2, array, status,        &
       alloc_status, deallocate_error_fatal, array_name, exact_size,           &
       bad_alloc, out )

!  Ensure that the 2D integer allocatable array "array" is of length at least
!  (len1,len2)

!  If exact_size is prsent and true, array is reallocated to be of size len.
!  Otherwise array is only reallocated if its length is currently smaller
!  than len

!  Dummy arguments

     INTEGER ( KIND = ip_ ), INTENT( IN ) :: len1, len2
     INTEGER ( KIND = ip_ ), INTENT( OUT ) :: status, alloc_status
     INTEGER ( KIND = ip_ ), ALLOCATABLE, DIMENSION( : , : ) :: array
     INTEGER ( KIND = ip_ ), OPTIONAL :: out
     LOGICAL, OPTIONAL :: deallocate_error_fatal, exact_size
     CHARACTER ( LEN = 80 ), OPTIONAL :: array_name
     CHARACTER ( LEN = 80 ), OPTIONAL :: bad_alloc

!  Local variable

     LOGICAL :: reallocate

!  Check to see if a reallocation (or initial allocation) is needed

     status = GALAHAD_ok ; alloc_status = 0 ; reallocate = .TRUE.
     IF ( PRESENT( bad_alloc ) ) bad_alloc = ''
     IF ( ALLOCATED( array ) ) THEN
       IF ( PRESENT( exact_size ) ) THEN
         IF ( exact_size ) THEN
           IF ( SIZE( array, 1 ) /= len1 .OR.                                  &
                SIZE( array, 2 ) /= len2  ) THEN
             CALL SPACE_dealloc_array( array, status, alloc_status,            &
                                       array_name, bad_alloc, out )
           ELSE ; reallocate = .FALSE.
           END IF
         ELSE
           IF ( SIZE( array, 1 ) /= len1 .OR.                                  &
                SIZE( array, 2 ) < len2  ) THEN
             CALL SPACE_dealloc_array( array, status, alloc_status,            &
                                       array_name, bad_alloc, out )
           ELSE ; reallocate = .FALSE.
           END IF
         END IF
       ELSE
         IF ( SIZE( array, 1 ) /= len1 .OR.                                    &
              SIZE( array, 2 ) < len2  ) THEN
           CALL SPACE_dealloc_array( array, status, alloc_status,              &
                                       array_name, bad_alloc, out )
          ELSE ; reallocate = .FALSE.
          END IF
       END IF
     END IF

!  If a deallocation error occured, return if desired

     IF ( PRESENT( deallocate_error_fatal ) ) THEN
       IF ( deallocate_error_fatal .AND. alloc_status /= 0 ) THEN
         status = GALAHAD_error_deallocate ; RETURN
       END IF
     ELSE
       IF ( alloc_status /= 0 ) THEN
         status = GALAHAD_error_deallocate ; RETURN
       END IF
     END IF

!  Reallocate array to be of length len1, len2, checking for error returns

     IF ( reallocate ) ALLOCATE( array( len1, len2 ), STAT = alloc_status )
     IF ( alloc_status /= 0 ) THEN
       status = GALAHAD_error_allocate
       IF ( PRESENT( bad_alloc ) .AND. PRESENT( array_name ) )                 &
         bad_alloc = array_name
       IF ( PRESENT( out ) ) THEN
         IF ( PRESENT( array_name ) ) THEN
           IF ( out > 0 ) WRITE( out, 2900 ) TRIM( array_name ), alloc_status
         ELSE
           IF ( out > 0 ) WRITE( out, 2910 ) alloc_status
         END IF
       END IF
     END IF
     RETURN

!  Non-executable statements

2900 FORMAT( ' ** Allocation error for ', A, /, '     status = ', I6 )
2910 FORMAT( ' ** Allocation error status = ', I6 )

!  End of SPACE_resize_integer2_array

     END SUBROUTINE SPACE_resize_integer2_array

!  -  S P A C E _ R E S I Z E _ L O G I C A L _ A R R A Y  S U B R O U T I N E

     SUBROUTINE SPACE_resize_logical_array( len, array, status, alloc_status,  &
       deallocate_error_fatal, array_name, exact_size, bad_alloc, out )

!  Ensure that the logical allocatable array "array" is of length at least len.

!  If exact_size is prsent and true, array is reallocated to be of size len.
!  Otherwise array is only reallocated if its length is currently smaller
!  than len

!  Dummy arguments

     INTEGER ( KIND = ip_ ), INTENT( IN ) :: len
     INTEGER ( KIND = ip_ ), INTENT( OUT ) :: status, alloc_status
     LOGICAL ( KIND = i4_ ), ALLOCATABLE, DIMENSION( : ) :: array
     INTEGER ( KIND = ip_ ), OPTIONAL :: out
     LOGICAL, OPTIONAL :: deallocate_error_fatal, exact_size
     CHARACTER ( LEN = 80 ), OPTIONAL :: array_name
     CHARACTER ( LEN = 80 ), OPTIONAL :: bad_alloc

!  Local variable

     LOGICAL :: reallocate

!  Check to see if a reallocation (or initial allocation) is needed

     status = GALAHAD_ok ; alloc_status = 0 ; reallocate = .TRUE.
     IF ( PRESENT( bad_alloc ) ) bad_alloc = ''
     IF ( ALLOCATED( array ) ) THEN
       IF ( PRESENT( exact_size ) ) THEN
         IF ( exact_size ) THEN
           IF ( SIZE( array ) /= len ) THEN
             CALL SPACE_dealloc_array( array, status, alloc_status,            &
                                       array_name, bad_alloc, out )
           ELSE ; reallocate = .FALSE.
           END IF
         ELSE
           IF ( SIZE( array ) < len ) THEN
             CALL SPACE_dealloc_array( array, status, alloc_status,            &
                                       array_name, bad_alloc, out )
           ELSE ; reallocate = .FALSE.
           END IF
         END IF
       ELSE
         IF ( SIZE( array ) < len ) THEN
           CALL SPACE_dealloc_array( array, status, alloc_status,             &
                                     array_name, bad_alloc, out )
          ELSE ; reallocate = .FALSE.
          END IF
       END IF
     END IF

!  If a deallocation error occured, return if desired

     IF ( PRESENT( deallocate_error_fatal ) ) THEN
       IF ( deallocate_error_fatal .AND. alloc_status /= 0 ) THEN
         status = GALAHAD_error_deallocate ; RETURN
       END IF
     ELSE
       IF ( alloc_status /= 0 ) THEN
         status = GALAHAD_error_deallocate ; RETURN
       END IF
     END IF

!  Reallocate array to be of length len, checking for error returns

     IF ( reallocate ) ALLOCATE( array( len ), STAT = alloc_status )
     IF ( alloc_status /= 0 ) THEN
       status = GALAHAD_error_allocate
       IF ( PRESENT( bad_alloc ) .AND. PRESENT( array_name ) )                 &
         bad_alloc = array_name
       IF ( PRESENT( out ) ) THEN
         IF ( PRESENT( array_name ) ) THEN
           IF ( out > 0 ) WRITE( out, 2900 ) TRIM( array_name ), alloc_status
         ELSE
           IF ( out > 0 ) WRITE( out, 2910 ) alloc_status
         END IF
       END IF
     END IF
     RETURN

!  Non-executable statements

2900 FORMAT( ' ** Allocation error for ', A, /, '     status = ', I6 )
2910 FORMAT( ' ** Allocation error status = ', I6 )

!  End of SPACE_resize_logical_array

     END SUBROUTINE SPACE_resize_logical_array

! - S P A C E _ R E S I Z E _ L O G I C A L 6 4 _ A R R A Y  S U B R O U T I N E

     SUBROUTINE SPACE_resize_logical64_array( len, array, status, alloc_status,&
       deallocate_error_fatal, array_name, exact_size, bad_alloc, out )

!  Ensure that the logical allocatable array "array" is of length at least len.

!  If exact_size is prsent and true, array is reallocated to be of size len.
!  Otherwise array is only reallocated if its length is currently smaller
!  than len

!  Dummy arguments

     INTEGER ( KIND = ip_ ), INTENT( IN ) :: len
     INTEGER ( KIND = ip_ ), INTENT( OUT ) :: status, alloc_status
     LOGICAL ( KIND = i8_ ), ALLOCATABLE, DIMENSION( : ) :: array
     INTEGER ( KIND = ip_ ), OPTIONAL :: out
     LOGICAL, OPTIONAL :: deallocate_error_fatal, exact_size
     CHARACTER ( LEN = 80 ), OPTIONAL :: array_name
     CHARACTER ( LEN = 80 ), OPTIONAL :: bad_alloc

!  Local variable

     LOGICAL :: reallocate

!  Check to see if a reallocation (or initial allocation) is needed

     status = GALAHAD_ok ; alloc_status = 0 ; reallocate = .TRUE.
     IF ( PRESENT( bad_alloc ) ) bad_alloc = ''
     IF ( ALLOCATED( array ) ) THEN
       IF ( PRESENT( exact_size ) ) THEN
         IF ( exact_size ) THEN
           IF ( SIZE( array ) /= len ) THEN
             CALL SPACE_dealloc_array( array, status, alloc_status,            &
                                       array_name, bad_alloc, out )
           ELSE ; reallocate = .FALSE.
           END IF
         ELSE
           IF ( SIZE( array ) < len ) THEN
             CALL SPACE_dealloc_array( array, status, alloc_status,            &
                                       array_name, bad_alloc, out )
           ELSE ; reallocate = .FALSE.
           END IF
         END IF
       ELSE
         IF ( SIZE( array ) < len ) THEN
           CALL SPACE_dealloc_array( array, status, alloc_status,             &
                                     array_name, bad_alloc, out )
          ELSE ; reallocate = .FALSE.
          END IF
       END IF
     END IF

!  If a deallocation error occured, return if desired

     IF ( PRESENT( deallocate_error_fatal ) ) THEN
       IF ( deallocate_error_fatal .AND. alloc_status /= 0 ) THEN
         status = GALAHAD_error_deallocate ; RETURN
       END IF
     ELSE
       IF ( alloc_status /= 0 ) THEN
         status = GALAHAD_error_deallocate ; RETURN
       END IF
     END IF

!  Reallocate array to be of length len, checking for error returns

     IF ( reallocate ) ALLOCATE( array( len ), STAT = alloc_status )
     IF ( alloc_status /= 0 ) THEN
       status = GALAHAD_error_allocate
       IF ( PRESENT( bad_alloc ) .AND. PRESENT( array_name ) )                 &
         bad_alloc = array_name
       IF ( PRESENT( out ) ) THEN
         IF ( PRESENT( array_name ) ) THEN
           IF ( out > 0 ) WRITE( out, 2900 ) TRIM( array_name ), alloc_status
         ELSE
           IF ( out > 0 ) WRITE( out, 2910 ) alloc_status
         END IF
       END IF
     END IF
     RETURN

!  Non-executable statements

2900 FORMAT( ' ** Allocation error for ', A, /, '     status = ', I6 )
2910 FORMAT( ' ** Allocation error status = ', I6 )

!  End of SPACE_resize_logical64_array

     END SUBROUTINE SPACE_resize_logical64_array

! - S P A C E _ R E S I Z E _ C H A R A C T E R _ A R R A Y  S U B R O U T I N E

     SUBROUTINE SPACE_resize_character_array( len, array, status,              &
       alloc_status, deallocate_error_fatal, array_name, exact_size,           &
       bad_alloc, out )

!  Ensure that the character allocatable array "array" is of length at least
!  len.

!  If exact_size is prsent and true, array is reallocated to be of size len.
!  Otherwise array is only reallocated if its length is currently smaller
!  than len

!  Dummy arguments

     INTEGER ( KIND = ip_ ), INTENT( IN ) :: len
     INTEGER ( KIND = ip_ ), INTENT( OUT ) :: status, alloc_status
     CHARACTER ( LEN = * ), ALLOCATABLE, DIMENSION( : ) :: array
     INTEGER ( KIND = ip_ ), OPTIONAL :: out
     LOGICAL, OPTIONAL :: deallocate_error_fatal, exact_size
     CHARACTER ( LEN = 80 ), OPTIONAL :: array_name
     CHARACTER ( LEN = 80 ), OPTIONAL :: bad_alloc

!  Local variable

     LOGICAL :: reallocate

!  Check to see if a reallocation (or initial allocation) is needed

     status = GALAHAD_ok ; alloc_status = 0 ; reallocate = .TRUE.
     IF ( PRESENT( bad_alloc ) ) bad_alloc = ''
     IF ( ALLOCATED( array ) ) THEN
       IF ( PRESENT( exact_size ) ) THEN
         IF ( exact_size ) THEN
           IF ( SIZE( array ) /= len ) THEN
             CALL SPACE_dealloc_array( array, status, alloc_status,           &
                                       array_name, bad_alloc, out )
           ELSE ; reallocate = .FALSE.
           END IF
         ELSE
           IF ( SIZE( array ) < len ) THEN
             CALL SPACE_dealloc_array( array, status, alloc_status,           &
                                       array_name, bad_alloc, out )
           ELSE ; reallocate = .FALSE.
           END IF
         END IF
       ELSE
         IF ( SIZE( array ) < len ) THEN
           CALL SPACE_dealloc_array( array, status, alloc_status,             &
                                     array_name, bad_alloc, out )
          ELSE ; reallocate = .FALSE.
          END IF
       END IF
     END IF

!  If a deallocation error occured, return if desired

     IF ( PRESENT( deallocate_error_fatal ) ) THEN
       IF ( deallocate_error_fatal .AND. alloc_status /= 0 ) THEN
         status = GALAHAD_error_deallocate ; RETURN
       END IF
     ELSE
       IF ( alloc_status /= 0 ) THEN
         status = GALAHAD_error_deallocate ; RETURN
       END IF
     END IF

!  Reallocate array to be of length len, checking for error returns

     IF ( reallocate ) ALLOCATE( array( len ), STAT = alloc_status )
     IF ( alloc_status /= 0 ) THEN
       status = GALAHAD_error_allocate
       IF ( PRESENT( bad_alloc ) .AND. PRESENT( array_name ) )                 &
         bad_alloc = array_name
       IF ( PRESENT( out ) ) THEN
         IF ( PRESENT( array_name ) ) THEN
           IF ( out > 0 ) WRITE( out, 2900 ) TRIM( array_name ), alloc_status
         ELSE
           IF ( out > 0 ) WRITE( out, 2910 ) alloc_status
         END IF
       END IF
     END IF
     RETURN

!  Non-executable statements

2900 FORMAT( ' ** Allocation error for ', A, /, '     status = ', I6 )
2910 FORMAT( ' ** Allocation error status = ', I6 )

!  End of SPACE_resize_character_array

     END SUBROUTINE SPACE_resize_character_array

! S P A C E _ R E S I Z E _ C H A R A C T E R 2 _ A R R A Y  S U B R O U T I N E

     SUBROUTINE SPACE_resize_character2_array( len1, len2, array, status,      &
       alloc_status, deallocate_error_fatal, array_name, exact_size,           &
       bad_alloc, out )

!  Ensure that the character allocatable array "array" is of length at least
!  (len1,len2)

!  If exact_size is prsent and true, array is reallocated to be of size len.
!  Otherwise array is only reallocated if its length is currently smaller
!  than len

!  Dummy arguments

     INTEGER ( KIND = ip_ ), INTENT( IN ) :: len1, len2
     INTEGER ( KIND = ip_ ), INTENT( OUT ) :: status, alloc_status
     CHARACTER ( LEN = * ), ALLOCATABLE, DIMENSION( : , : ) :: array
     INTEGER ( KIND = ip_ ), OPTIONAL :: out
     LOGICAL, OPTIONAL :: deallocate_error_fatal, exact_size
     CHARACTER ( LEN = 80 ), OPTIONAL :: array_name
     CHARACTER ( LEN = 80 ), OPTIONAL :: bad_alloc

!  Local variable

     LOGICAL :: reallocate

!  Check to see if a reallocation (or initial allocation) is needed

     status = GALAHAD_ok ; alloc_status = 0 ; reallocate = .TRUE.
     IF ( PRESENT( bad_alloc ) ) bad_alloc = ''
     IF ( ALLOCATED( array ) ) THEN
       IF ( PRESENT( exact_size ) ) THEN
         IF ( exact_size ) THEN
           IF ( SIZE( array, 1 ) /= len1 .OR.                                  &
                SIZE( array, 2 ) /= len2  ) THEN
             CALL SPACE_dealloc_array( array, status, alloc_status,            &
                                       array_name, bad_alloc, out )
           ELSE ; reallocate = .FALSE.
           END IF
         ELSE
           IF ( SIZE( array, 1 ) /= len1 .OR.                                  &
                SIZE( array, 2 ) < len2  ) THEN
             CALL SPACE_dealloc_array( array, status, alloc_status,            &
                                       array_name, bad_alloc, out )
           ELSE ; reallocate = .FALSE.
           END IF
         END IF
       ELSE
         IF ( SIZE( array, 1 ) /= len1 .OR.                                    &
              SIZE( array, 2 ) < len2  ) THEN
           CALL SPACE_dealloc_array( array, status, alloc_status,              &
                                     array_name, bad_alloc, out )
          ELSE ; reallocate = .FALSE.
          END IF
       END IF
     END IF

!  If a deallocation error occured, return if desired

     IF ( PRESENT( deallocate_error_fatal ) ) THEN
       IF ( deallocate_error_fatal .AND. alloc_status /= 0 ) THEN
         status = GALAHAD_error_deallocate ; RETURN
       END IF
     ELSE
       IF ( alloc_status /= 0 ) THEN
         status = GALAHAD_error_deallocate ; RETURN
       END IF
     END IF

!  Reallocate array to be of length len1, len2, checking for error returns

     IF ( reallocate ) ALLOCATE( array( len1, len2 ), STAT = alloc_status )
     IF ( alloc_status /= 0 ) THEN
       status = GALAHAD_error_allocate
       IF ( PRESENT( bad_alloc ) .AND. PRESENT( array_name ) )                 &
         bad_alloc = array_name
       IF ( PRESENT( out ) ) THEN
         IF ( PRESENT( array_name ) ) THEN
           IF ( out > 0 ) WRITE( out, 2900 ) TRIM( array_name ), alloc_status
         ELSE
           IF ( out > 0 ) WRITE( out, 2910 ) alloc_status
         END IF
       END IF
     END IF
     RETURN

!  Non-executable statements

2900 FORMAT( ' ** Allocation error for ', A, /, '     status = ', I6 )
2910 FORMAT( ' ** Allocation error status = ', I6 )

!  End of SPACE_resize_character2_array

     END SUBROUTINE SPACE_resize_character2_array

!  *-  S P A C E _ R E S I Z E _ R E A L _ C A R R A Y  S U B R O U T I N E   -*

     SUBROUTINE SPACE_resize_real_carray( len, array, status, alloc_status,    &
       deallocate_error_fatal, array_name, exact_size, bad_alloc, out )

!  Ensure that the real allocatable c-style array "array" is of length at
!  least len.

!  If exact_size is prsent and true, array is reallocated to be of size len.
!  Otherwise array is only reallocated if its length is currently smaller
!  than len

!  Dummy arguments

     INTEGER ( KIND = ip_ ), INTENT( IN ) :: len
     INTEGER ( KIND = ip_ ), INTENT( OUT ) :: status, alloc_status
     REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : ) :: array
     INTEGER ( KIND = ip_ ), OPTIONAL :: out
     LOGICAL, OPTIONAL :: deallocate_error_fatal, exact_size
     CHARACTER ( LEN = 80 ), OPTIONAL :: array_name
     CHARACTER ( LEN = 80 ), OPTIONAL :: bad_alloc

!  Local variable

     LOGICAL :: reallocate

!  Check to see if a reallocation (or initial allocation) is needed

     status = GALAHAD_ok ; alloc_status = 0 ; reallocate = .TRUE.
     IF ( PRESENT( bad_alloc ) ) bad_alloc = ''
     IF ( ALLOCATED( array ) ) THEN
       IF ( PRESENT( exact_size ) ) THEN
         IF ( exact_size ) THEN
           IF ( LBOUND( array, 1 ) /= 0 .OR.                                   &
                UBOUND( array, 1 ) /= len - 1 ) THEN
             CALL SPACE_dealloc_array( array, status, alloc_status,            &
                                       array_name, bad_alloc, out )
           ELSE ; reallocate = .FALSE.
           END IF
         ELSE
           IF ( LBOUND( array, 1 ) /= 0 .OR.                                   &
                UBOUND( array, 1 ) < len - 1 ) THEN
             CALL SPACE_dealloc_array( array, status, alloc_status,            &
                                       array_name, bad_alloc, out )
           ELSE ; reallocate = .FALSE.
           END IF
         END IF
       ELSE
         IF ( LBOUND( array, 1 ) /= 0 .OR.                                     &
              UBOUND( array, 1 ) < len - 1 ) THEN
           CALL SPACE_dealloc_array( array, status, alloc_status,              &
                                     array_name, bad_alloc, out )
          ELSE ; reallocate = .FALSE.
          END IF
       END IF
     END IF

!  If a deallocation error occured, return if desired

     IF ( PRESENT( deallocate_error_fatal ) ) THEN
       IF ( deallocate_error_fatal .AND. alloc_status /= 0 ) THEN
         status = GALAHAD_error_deallocate ; RETURN
       END IF
     ELSE
       IF ( alloc_status /= 0 ) THEN
         status = GALAHAD_error_deallocate ; RETURN
       END IF
     END IF

!  Reallocate array to be of length len, checking for error returns

     IF ( reallocate ) ALLOCATE( array( 0 : len - 1 ), STAT = alloc_status )
     IF ( alloc_status /= 0 ) THEN
       status = GALAHAD_error_allocate
       IF ( PRESENT( bad_alloc ) .AND. PRESENT( array_name ) )                 &
         bad_alloc = array_name
       IF ( PRESENT( out ) ) THEN
         IF ( PRESENT( array_name ) ) THEN
           IF ( out > 0 ) WRITE( out, 2900 ) TRIM( array_name ), alloc_status
         ELSE
           IF ( out > 0 ) WRITE( out, 2910 ) alloc_status
         END IF
       END IF
     END IF
     RETURN

!  Non-executable statements

2900 FORMAT( ' ** Allocation error for ', A, /, '     status = ', I6 )
2910 FORMAT( ' ** Allocation error status = ', I6 )

!  End of SPACE_resize_real_carray

     END SUBROUTINE SPACE_resize_real_carray

!  - S P A C E _ R E S I Z E _ I N T E G E R _ C A R R A Y  S U B R O U T I N E

     SUBROUTINE SPACE_resize_integer_carray( len, array, status, alloc_status, &
       deallocate_error_fatal, array_name, exact_size, bad_alloc, out )

!  Ensure that the integer allocatable c-style array "array" is of length at
!  least len.

!  If exact_size is prsent and true, array is reallocated to be of size len.
!  Otherwise array is only reallocated if its length is currently smaller
!  than len

!  Dummy arguments

     INTEGER ( KIND = ip_ ), INTENT( IN ) :: len
     INTEGER ( KIND = ip_ ), INTENT( OUT ) :: status, alloc_status
     INTEGER ( KIND = ip_ ), ALLOCATABLE, DIMENSION( : ) :: array
     INTEGER ( KIND = ip_ ), OPTIONAL :: out
     LOGICAL, OPTIONAL :: deallocate_error_fatal, exact_size
     CHARACTER ( LEN = 80 ), OPTIONAL :: array_name
     CHARACTER ( LEN = 80 ), OPTIONAL :: bad_alloc

!  Local variable

     LOGICAL :: reallocate

!  Check to see if a reallocation (or initial allocation) is needed

     status = GALAHAD_ok ; alloc_status = 0 ; reallocate = .TRUE.
     IF ( PRESENT( bad_alloc ) ) bad_alloc = ''
     IF ( ALLOCATED( array ) ) THEN
       IF ( PRESENT( exact_size ) ) THEN
         IF ( exact_size ) THEN
           IF ( LBOUND( array, 1 ) /= 0 .OR.                                   &
                UBOUND( array, 1 ) /= len - 1 ) THEN
             CALL SPACE_dealloc_array( array, status, alloc_status,            &
                                       array_name, bad_alloc, out )
           ELSE ; reallocate = .FALSE.
           END IF
         ELSE
           IF ( LBOUND( array, 1 ) /= 0 .OR.                                   &
                UBOUND( array, 1 ) < len - 1 ) THEN
             CALL SPACE_dealloc_array( array, status, alloc_status,            &
                                       array_name, bad_alloc, out )
           ELSE ; reallocate = .FALSE.
           END IF
         END IF
       ELSE
         IF ( LBOUND( array, 1 ) /= 0 .OR.                                     &
              UBOUND( array, 1 ) < len - 1 ) THEN
           CALL SPACE_dealloc_array( array, status, alloc_status,              &
                                     array_name, bad_alloc, out )
          ELSE ; reallocate = .FALSE.
          END IF
       END IF
     END IF

!  If a deallocation error occured, return if desired

     IF ( PRESENT( deallocate_error_fatal ) ) THEN
       IF ( deallocate_error_fatal .AND. alloc_status /= 0 ) THEN
         status = GALAHAD_error_deallocate ; RETURN
       END IF
     ELSE
       IF ( alloc_status /= 0 ) THEN
         status = GALAHAD_error_deallocate ; RETURN
       END IF
     END IF

!  Reallocate array to be of length len, checking for error returns

     IF ( reallocate ) ALLOCATE( array( 0 : len - 1 ), STAT = alloc_status )
     IF ( alloc_status /= 0 ) THEN
       status = GALAHAD_error_allocate
       IF ( PRESENT( bad_alloc ) .AND. PRESENT( array_name ) )                 &
         bad_alloc = array_name
       IF ( PRESENT( out ) ) THEN
         IF ( PRESENT( array_name ) ) THEN
           IF ( out > 0 ) WRITE( out, 2900 ) TRIM( array_name ), alloc_status
         ELSE
           IF ( out > 0 ) WRITE( out, 2910 ) alloc_status
         END IF
       END IF
     END IF
     RETURN

!  Non-executable statements

2900 FORMAT( ' ** Allocation error for ', A, /, '     status = ', I6 )
2910 FORMAT( ' ** Allocation error status = ', I6 )

!  End of SPACE_resize_integer_carray

     END SUBROUTINE SPACE_resize_integer_carray

!!$! -S P A C E _ R E S I Z E _ L O G I C A L _ C A R R A Y  S U B R O U T I N E
!!$
!!$   SUBROUTINE SPACE_resize_logical_carray( len, array, status, alloc_status,&
!!$       deallocate_error_fatal, array_name, exact_size, bad_alloc, out )
!!$
!!$!  Ensure that the logical allocatable c-style array "array" is of length at
!!$!  least len.
!!$
!!$!  If exact_size is prsent and true, array is reallocated to be of size len.
!!$!  Otherwise array is only reallocated if its length is currently smaller
!!$!  than len
!!$
!!$!  Dummy arguments
!!$
!!$     INTEGER ( KIND = ip_ ), INTENT( IN ) :: len
!!$     INTEGER ( KIND = ip_ ), INTENT( OUT ) :: status, alloc_status
!!$     LOGICAL ( KIND = lp_ ), ALLOCATABLE, DIMENSION( : ) :: array
!!$     INTEGER ( KIND = ip_ ), OPTIONAL :: out
!!$     LOGICAL, OPTIONAL :: deallocate_error_fatal, exact_size
!!$     CHARACTER ( LEN = 80 ), OPTIONAL :: array_name
!!$     CHARACTER ( LEN = 80 ), OPTIONAL :: bad_alloc
!!$
!!$!  Local variable
!!$
!!$     LOGICAL :: reallocate
!!$
!!$!  Check to see if a reallocation (or initial allocation) is needed
!!$
!!$     status = GALAHAD_ok ; alloc_status = 0 ; reallocate = .TRUE.
!!$     IF ( PRESENT( bad_alloc ) ) bad_alloc = ''
!!$     IF ( ALLOCATED( array ) ) THEN
!!$       IF ( PRESENT( exact_size ) ) THEN
!!$         IF ( exact_size ) THEN
!!$           IF ( LBOUND( array, 1 ) /= 0 .OR.                                &
!!$                UBOUND( array, 1 ) /= len - 1 ) THEN
!!$             CALL SPACE_dealloc_array( array, status, alloc_status,         &
!!$                                       array_name, bad_alloc, out )
!!$           ELSE ; reallocate = .FALSE.
!!$           END IF
!!$         ELSE
!!$           IF ( LBOUND( array, 1 ) /= 0 .OR.                                &
!!$                UBOUND( array, 1 ) < len - 1 ) THEN
!!$             CALL SPACE_dealloc_array( array, status, alloc_status,         &
!!$                                       array_name, bad_alloc, out )
!!$           ELSE ; reallocate = .FALSE.
!!$           END IF
!!$         END IF
!!$       ELSE
!!$         IF ( LBOUND( array, 1 ) /= 0 .OR.                                  &
!!$              UBOUND( array, 1 ) < len - 1 ) THEN
!!$           CALL SPACE_dealloc_array( array, status, alloc_status,           &
!!$                                     array_name, bad_alloc, out )
!!$          ELSE ; reallocate = .FALSE.
!!$          END IF
!!$       END IF
!!$     END IF
!!$
!!$!  If a deallocation error occured, return if desired
!!$
!!$     IF ( PRESENT( deallocate_error_fatal ) ) THEN
!!$       IF ( deallocate_error_fatal .AND. alloc_status /= 0 ) THEN
!!$         status = GALAHAD_error_deallocate ; RETURN
!!$       END IF
!!$     ELSE
!!$       IF ( alloc_status /= 0 ) THEN
!!$         status = GALAHAD_error_deallocate ; RETURN
!!$       END IF
!!$     END IF
!!$
!!$!  Reallocate array to be of length len, checking for error returns
!!$
!!$     IF ( reallocate ) ALLOCATE( array( 0 : len - 1 ), STAT = alloc_status )
!!$     IF ( alloc_status /= 0 ) THEN
!!$       status = GALAHAD_error_allocate
!!$       IF ( PRESENT( bad_alloc ) .AND. PRESENT( array_name ) )              &
!!$         bad_alloc = array_name
!!$       IF ( PRESENT( out ) ) THEN
!!$         IF ( PRESENT( array_name ) ) THEN
!!$           IF ( out > 0 ) WRITE( out, 2900 ) TRIM( array_name ), alloc_status
!!$         ELSE
!!$           IF ( out > 0 ) WRITE( out, 2910 ) alloc_status
!!$         END IF
!!$       END IF
!!$     END IF
!!$     RETURN
!!$
!!$!  Non-executable statements
!!$
!!$2900 FORMAT( ' ** Allocation error for ', A, /, '     status = ', I6 )
!!$2910 FORMAT( ' ** Allocation error status = ', I6 )
!!$
!!$!  End of SPACE_resize_logical_carray
!!$
!!$     END SUBROUTINE SPACE_resize_logical_carray

!-  S P A C E _ e x t e n d _ a r r a y _ i n t e g e r  S U B R O U T I N E -

     SUBROUTINE SPACE_extend_array_integer( ARRAY, old_length, used_length,    &
                                            new_length, min_length, buffer,    &
                                            status, alloc_status )

!  -------------------------------------------------------------------------
!  extend an integer array so that its length is increaed from old_length to
!  as close to new_length as possible while keeping existing data intact
!  -------------------------------------------------------------------------

!  History -
!   fortran 90 version released pre GALAHAD Version 1.0. February 7th 1995 as
!     EXTEND_array_integer as part of the GALAHAD module EXTEND
!   fortran 2003 version released in SIFDECODE/CUTEst, 5th November 2012
!   borrowed for GALAHAD, 12th May 2013

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

!  make sure that the new length is larger than the old

     IF ( new_length <= old_length ) new_length = 2 * old_length

!  ensure that the input data is consistent

     used_length = MIN( used_length, old_length )
     min_length = MAX( old_length + 1, MIN( min_length, new_length ) )

!  if possible, allocate DUMMY to hold the old values of ARRAY

     ALLOCATE( DUMMY( used_length ), STAT = alloc_status )

!  if the allocation failed, resort to using an external unit

     IF ( alloc_status /= 0 ) GO TO 100

     DUMMY( : used_length ) = ARRAY( : used_length )

!  extend the length of ARRAY

     DEALLOCATE( ARRAY, STAT = alloc_status )
     IF ( alloc_status /= 0 ) THEN
       status = GALAHAD_error_deallocate
       RETURN
     END IF
     length = new_length

  10 CONTINUE
     ALLOCATE( ARRAY( length ), STAT = alloc_status )

!  if the allocation failed, reduce the new length and retry

     IF ( alloc_status /= 0 ) THEN
       length = length + ( length - min_length ) / 2

!  if there is insufficient room for both ARRAY and DUMMY, use an external unit

       IF ( length < min_length ) THEN

!  rewind the buffer i/o unit

         INQUIRE( UNIT = buffer, OPENED = file_open )
         IF ( file_open ) THEN
           REWIND( UNIT = buffer )
         ELSE
           OPEN( UNIT = buffer )
         END IF

!  copy the contents of ARRAY into the buffer i/o area

         WRITE( UNIT = buffer, FMT = * ) DUMMY( : used_length )

!  extend the length of ARRAY

         DEALLOCATE( DUMMY )
         GO TO 110
       END IF
       GO TO 10
     END IF

!  copy the contents of ARRAY back from the buffer i/o area

     ARRAY( : used_length ) = DUMMY( : used_length )
     DEALLOCATE( DUMMY )
     new_length = length
     GO TO 200

!  use an external unit for writing

 100 CONTINUE

!  rewind the buffer i/o unit

     INQUIRE( UNIT = buffer, OPENED = file_open )
     IF ( file_open ) THEN
       REWIND( UNIT = buffer )
     ELSE
       OPEN( UNIT = buffer )
     END IF

!  copy the contents of ARRAY into the buffer i/o area

     WRITE( UNIT = buffer, FMT = * ) ARRAY( : used_length )

!  extend the length of ARRAY

     DEALLOCATE( ARRAY )

 110 CONTINUE
     ALLOCATE( ARRAY( new_length ), STAT = alloc_status )

!  if the allocation failed, reduce the new length and retry

     IF ( alloc_status /= 0 ) THEN
       new_length = min_length + ( new_length - min_length ) / 2
       IF ( new_length < min_length ) THEN
         status = GALAHAD_error_allocate
         RETURN
       END IF
       GO TO 110
     END IF

!  copy the contents of ARRAY back from the buffer i/o area

     REWIND( UNIT = buffer )
     READ( UNIT = buffer, FMT = * ) ARRAY( : used_length )

!  successful exit

 200 CONTINUE
     status = GALAHAD_ok
     RETURN

!  end of subroutine SPACE_extend_array_integer

     END SUBROUTINE SPACE_extend_array_integer

!-*-*-  S P A C E _ e x t e n d _ a r r a y _ r e a l  S U B R O U T I N E -*-*-

     SUBROUTINE SPACE_extend_array_real( ARRAY, old_length, used_length,       &
                                         new_length, min_length, buffer,       &
                                         status, alloc_status )

!  ---------------------------------------------------------------------
!  extend a real array so that its length is increaed from old_length to
!  as close to new_length as possible while keeping existing data intact
!  ---------------------------------------------------------------------

!  History -
!   fortran 90 version released pre GALAHAD Version 1.0. February 7th 1995 as
!     EXTEND_array_real as part of the GALAHAD module EXTEND
!   fortran 2003 version released in SIFDECODE/CUTEst, 5th November 2012
!   borrowed for GALAHAD, 12th May 2013

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     INTEGER ( KIND = ip_ ), INTENT( IN ) :: old_length, buffer
     INTEGER ( KIND = ip_ ), INTENT( OUT ) :: status, alloc_status
     INTEGER ( KIND = ip_ ), INTENT( INOUT ) :: used_length, min_length
     INTEGER ( KIND = ip_ ), INTENT( INOUT ) :: new_length
     REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : ) :: ARRAY

     INTEGER ( KIND = ip_ ) :: length
     LOGICAL :: file_open
     REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : ) :: DUMMY

!  make sure that the new length is larger than the old

     IF ( new_length <= old_length ) new_length = 2 * old_length

!  ensure that the input data is consistent

     used_length = MIN( used_length, old_length )
     min_length = MAX( old_length + 1, MIN( min_length, new_length ) )

!  if possible, allocate DUMMY to hold the old values of ARRAY

     ALLOCATE( DUMMY( used_length ), STAT = alloc_status )

!  if the allocation failed, resort to using an external unit

     IF ( alloc_status /= 0 ) GO TO 100

     DUMMY( : used_length ) = ARRAY( : used_length )

!  extend the length of ARRAY

     DEALLOCATE( ARRAY, STAT = alloc_status )
     IF ( alloc_status /= 0 ) THEN
       status = GALAHAD_error_deallocate
       RETURN
     END IF
     length = new_length

  10 CONTINUE
     ALLOCATE( ARRAY( length ), STAT = alloc_status )

!  if the allocation failed, reduce the new length and retry

     IF ( alloc_status /= 0 ) THEN
       length = length + ( length - min_length ) / 2

!  if there is insufficient room for both ARRAY and DUMMY, use an external unit

       IF ( length < min_length ) THEN

!  rewind the buffer i/o unit

         INQUIRE( UNIT = buffer, OPENED = file_open )
         IF ( file_open ) THEN
           REWIND( UNIT = buffer )
         ELSE
           OPEN( UNIT = buffer )
         END IF

!  copy the contents of ARRAY into the buffer i/o area

         WRITE( UNIT = buffer, FMT = * ) DUMMY( : used_length )

!  extend the length of ARRAY

         DEALLOCATE( DUMMY )
         GO TO 110
       END IF
       GO TO 10
     END IF

!  copy the contents of ARRAY back from the buffer i/o area

       ARRAY( : used_length ) = DUMMY( : used_length )
       DEALLOCATE( DUMMY )
       new_length = length
       GO TO 200

!  use an external unit for writing

 100   CONTINUE

!  rewind the buffer i/o unit

     INQUIRE( UNIT = buffer, OPENED = file_open )
     IF ( file_open ) THEN
       REWIND( UNIT = buffer )
     ELSE
       OPEN( UNIT = buffer )
     END IF

!  copy the contents of ARRAY into the buffer i/o area

     WRITE( UNIT = buffer, FMT = * ) ARRAY( : used_length )

!  extend the length of ARRAY

     DEALLOCATE( ARRAY )

 110 CONTINUE
     ALLOCATE( ARRAY( new_length ), STAT = alloc_status )

!  if the allocation failed, reduce the new length and retry

     IF ( alloc_status /= 0 ) THEN
       new_length = min_length + ( new_length - min_length ) / 2
       IF ( new_length < min_length ) THEN
          status = GALAHAD_error_allocate
          RETURN
       END IF
       GO TO 110
     END IF

!  copy the contents of ARRAY back from the buffer i/o area

     REWIND( UNIT = buffer )
     READ( UNIT = buffer, FMT = * ) ARRAY( : used_length )

!  successful exit

 200 CONTINUE
     status = GALAHAD_ok
     RETURN

!  end of subroutine SPACE_extend_array_real

     END SUBROUTINE SPACE_extend_array_real

!-  S P A C E _ e x t e n d _ a r r a y _ l o g i c a l   S U B R O U T I N E -

     SUBROUTINE SPACE_extend_array_logical( ARRAY, old_length, used_length,    &
                                            new_length, min_length, buffer,    &
                                            status, alloc_status )

!  -------------------------------------------------------------------------
!  extend an integer array so that its length is increaed from old_length to
!  as close to new_length as possible while keeping existing data intact
!  -------------------------------------------------------------------------

!  History -
!   fortran 90 version released pre GALAHAD Version 1.0. February 7th 1995 as
!     EXTEND_array_integer as part of the GALAHAD module EXTEND
!   fortran 2003 version released in SIFDECODE/CUTEst, 5th November 2012
!   borrowed for GALAHAD, 12th May 2013

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     INTEGER ( KIND = ip_ ), INTENT( IN ) :: old_length, buffer
     INTEGER ( KIND = ip_ ), INTENT( OUT ) :: status, alloc_status
     INTEGER ( KIND = ip_ ), INTENT( INOUT ) :: used_length, min_length
     INTEGER ( KIND = ip_ ), INTENT( INOUT ) :: new_length
     LOGICAL, ALLOCATABLE, DIMENSION( : ) :: ARRAY

     INTEGER ( KIND = ip_ ) :: length
     LOGICAL :: file_open
     LOGICAL, ALLOCATABLE, DIMENSION( : ) :: DUMMY

!  make sure that the new length is larger than the old

     IF ( new_length <= old_length ) new_length = 2 * old_length

!  ensure that the input data is consistent

     used_length = MIN( used_length, old_length )
     min_length = MAX( old_length + 1, MIN( min_length, new_length ) )

!  if possible, allocate DUMMY to hold the old values of ARRAY

     ALLOCATE( DUMMY( used_length ), STAT = alloc_status )

!  if the allocation failed, resort to using an external unit

     IF ( alloc_status /= 0 ) GO TO 100

     DUMMY( : used_length ) = ARRAY( : used_length )

!  extend the length of ARRAY

     DEALLOCATE( ARRAY, STAT = alloc_status )
     IF ( alloc_status /= 0 ) THEN
       status = GALAHAD_error_deallocate
       RETURN
     END IF
     length = new_length

  10 CONTINUE
     ALLOCATE( ARRAY( length ), STAT = alloc_status )

!  if the allocation failed, reduce the new length and retry

     IF ( alloc_status /= 0 ) THEN
       length = length + ( length - min_length ) / 2

!  if there is insufficient room for both ARRAY and DUMMY, use an external unit

       IF ( length < min_length ) THEN

!  rewind the buffer i/o unit

         INQUIRE( UNIT = buffer, OPENED = file_open )
         IF ( file_open ) THEN
           REWIND( UNIT = buffer )
         ELSE
           OPEN( UNIT = buffer )
         END IF

!  copy the contents of ARRAY into the buffer i/o area

         WRITE( UNIT = buffer, FMT = * ) DUMMY( : used_length )

!  extend the length of ARRAY

         DEALLOCATE( DUMMY )
         GO TO 110
       END IF
       GO TO 10
     END IF

!  copy the contents of ARRAY back from the buffer i/o area

     ARRAY( : used_length ) = DUMMY( : used_length )
     DEALLOCATE( DUMMY )
     new_length = length
     GO TO 200

!  use an external unit for writing

 100 CONTINUE

!  rewind the buffer i/o unit

     INQUIRE( UNIT = buffer, OPENED = file_open )
     IF ( file_open ) THEN
       REWIND( UNIT = buffer )
     ELSE
       OPEN( UNIT = buffer )
     END IF

!  copy the contents of ARRAY into the buffer i/o area

     WRITE( UNIT = buffer, FMT = * ) ARRAY( : used_length )

!  extend the length of ARRAY

     DEALLOCATE( ARRAY )

 110 CONTINUE
     ALLOCATE( ARRAY( new_length ), STAT = alloc_status )

!  if the allocation failed, reduce the new length and retry

     IF ( alloc_status /= 0 ) THEN
       new_length = min_length + ( new_length - min_length ) / 2
       IF ( new_length < min_length ) THEN
         status = GALAHAD_error_allocate
         RETURN
       END IF
       GO TO 110
     END IF

!  copy the contents of ARRAY back from the buffer i/o area

     REWIND( UNIT = buffer )
     READ( UNIT = buffer, FMT = * ) ARRAY( : used_length )

!  successful exit

 200 CONTINUE
     status = GALAHAD_ok
     RETURN

!  end of subroutine SPACE_extend_array_logical

     END SUBROUTINE SPACE_extend_array_logical

!-  S P A C E _ e x t e n d _ c a r r a y _ i n t e g e r  S U B R O U T I N E -

     SUBROUTINE SPACE_extend_carray_integer( ARRAY, old_length, used_length,   &
                                             new_length, min_length, buffer,   &
                                             status, alloc_status )

!  -------------------------------------------------------------------------
!  extend an integer c-style array so that its length is increaed from
!  old_length to as close to new_length as possible while keeping existing
!  data intact
!  -------------------------------------------------------------------------

!  History -
!   fortran 90 version released pre GALAHAD Version 1.0. February 7th 1995 as
!     EXTEND_array_integer as part of the GALAHAD module EXTEND
!   fortran 2003 version released in SIFDECODE/CUTEst, 5th November 2012
!   borrowed for GALAHAD, and extended for c-style arrays 15th May 2014

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

!  make sure that the new length is larger than the old

     IF ( new_length <= old_length ) new_length = 2 * old_length

!  ensure that the input data is consistent

     used_length = MIN( used_length, old_length )
     min_length = MAX( old_length + 1, MIN( min_length, new_length ) )

!  if possible, allocate DUMMY to hold the old values of ARRAY

     ALLOCATE( DUMMY( 0 : used_length - 1 ), STAT = alloc_status )

!  if the allocation failed, resort to using an external unit

     IF ( alloc_status /= 0 ) GO TO 100

     DUMMY( 0 : used_length - 1 ) = ARRAY( 0 : used_length - 1 )

!  extend the length of ARRAY

     DEALLOCATE( ARRAY, STAT = alloc_status )
     IF ( alloc_status /= 0 ) THEN
       status = GALAHAD_error_deallocate
       RETURN
     END IF
     length = new_length

  10 CONTINUE
     ALLOCATE( ARRAY( 0 : length - 1 ), STAT = alloc_status )

!  if the allocation failed, reduce the new length and retry

     IF ( alloc_status /= 0 ) THEN
       length = length + ( length - min_length ) / 2

!  if there is insufficient room for both ARRAY and DUMMY, use an external unit

       IF ( length < min_length ) THEN

!  rewind the buffer i/o unit

         INQUIRE( UNIT = buffer, OPENED = file_open )
         IF ( file_open ) THEN
           REWIND( UNIT = buffer )
         ELSE
           OPEN( UNIT = buffer )
         END IF

!  copy the contents of ARRAY into the buffer i/o area

         WRITE( UNIT = buffer, FMT = * ) DUMMY( 0 : used_length - 1 )

!  extend the length of ARRAY

         DEALLOCATE( DUMMY )
         GO TO 110
       END IF
       GO TO 10
     END IF

!  copy the contents of ARRAY back from the buffer i/o area

     ARRAY( 0 : used_length - 1 ) = DUMMY( 0 : used_length - 1 )
     DEALLOCATE( DUMMY )
     new_length = length
     GO TO 200

!  use an external unit for writing

 100 CONTINUE

!  rewind the buffer i/o unit

     INQUIRE( UNIT = buffer, OPENED = file_open )
     IF ( file_open ) THEN
       REWIND( UNIT = buffer )
     ELSE
       OPEN( UNIT = buffer )
     END IF

!  copy the contents of ARRAY into the buffer i/o area

     WRITE( UNIT = buffer, FMT = * ) ARRAY( 0 : used_length - 1 )

!  extend the length of ARRAY

     DEALLOCATE( ARRAY )

 110 CONTINUE
     ALLOCATE( ARRAY( 0 : new_length - 1 ), STAT = alloc_status )

!  if the allocation failed, reduce the new length and retry

     IF ( alloc_status /= 0 ) THEN
       new_length = min_length + ( new_length - min_length ) / 2
       IF ( new_length < min_length ) THEN
         status = GALAHAD_error_allocate
         RETURN
       END IF
       GO TO 110
     END IF

!  copy the contents of ARRAY back from the buffer i/o area

     REWIND( UNIT = buffer )
     READ( UNIT = buffer, FMT = * ) ARRAY( 0 : used_length - 1 )

!  successful exit

 200 CONTINUE
     status = GALAHAD_ok
     RETURN

!  end of subroutine SPACE_extend_carray_integer

     END SUBROUTINE SPACE_extend_carray_integer

!-*-  S P A C E _ e x t e n d _ c a r r a y _ r e a l  S U B R O U T I N E -*-

     SUBROUTINE SPACE_extend_carray_real( ARRAY, old_length, used_length,      &
                                          new_length, min_length, buffer,      &
                                          status, alloc_status )

!  ---------------------------------------------------------------------
!  extend a real c-style array so that its length is increaed from old_length
!  to as close to new_length as possible while keeping existing data intact
!  ---------------------------------------------------------------------

!  History -
!   fortran 90 version released pre GALAHAD Version 1.0. February 7th 1995 as
!     EXTEND_array_real as part of the GALAHAD module EXTEND
!   fortran 2003 version released in SIFDECODE/CUTEst, 5th November 2012
!   borrowed for GALAHAD, and extended for c-style arrays 15th May 2014

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     INTEGER ( KIND = ip_ ), INTENT( IN ) :: old_length, buffer
     INTEGER ( KIND = ip_ ), INTENT( OUT ) :: status, alloc_status
     INTEGER ( KIND = ip_ ), INTENT( INOUT ) :: used_length, min_length
     INTEGER ( KIND = ip_ ), INTENT( INOUT ) :: new_length
     REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : ) :: ARRAY

     INTEGER ( KIND = ip_ ) :: length
     LOGICAL :: file_open
     REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : ) :: DUMMY

!  make sure that the new length is larger than the old

     IF ( new_length <= old_length ) new_length = 2 * old_length

!  ensure that the input data is consistent

     used_length = MIN( used_length, old_length )
     min_length = MAX( old_length + 1, MIN( min_length, new_length ) )

!  if possible, allocate DUMMY to hold the old values of ARRAY

     ALLOCATE( DUMMY( 0 : used_length - 1 ), STAT = alloc_status )

!  if the allocation failed, resort to using an external unit

     IF ( alloc_status /= 0 ) GO TO 100

     DUMMY( 0 : used_length - 1 ) = ARRAY( 0 : used_length - 1 )

!  extend the length of ARRAY

     DEALLOCATE( ARRAY, STAT = alloc_status )
     IF ( alloc_status /= 0 ) THEN
       status = GALAHAD_error_deallocate
       RETURN
     END IF
     length = new_length

  10 CONTINUE
     ALLOCATE( ARRAY( 0 : length - 1 ), STAT = alloc_status )

!  if the allocation failed, reduce the new length and retry

     IF ( alloc_status /= 0 ) THEN
       length = length + ( length - min_length ) / 2

!  if there is insufficient room for both ARRAY and DUMMY, use an external unit

       IF ( length < min_length ) THEN

!  rewind the buffer i/o unit

         INQUIRE( UNIT = buffer, OPENED = file_open )
         IF ( file_open ) THEN
           REWIND( UNIT = buffer )
         ELSE
           OPEN( UNIT = buffer )
         END IF

!  copy the contents of ARRAY into the buffer i/o area

         WRITE( UNIT = buffer, FMT = * ) DUMMY( 0 : used_length - 1 )

!  extend the length of ARRAY

         DEALLOCATE( DUMMY )
         GO TO 110
       END IF
       GO TO 10
     END IF

!  copy the contents of ARRAY back from the buffer i/o area

       ARRAY( 0 : used_length - 1 ) = DUMMY( 0 : used_length - 1 )
       DEALLOCATE( DUMMY )
       new_length = length
       GO TO 200

!  use an external unit for writing

 100   CONTINUE

!  rewind the buffer i/o unit

     INQUIRE( UNIT = buffer, OPENED = file_open )
     IF ( file_open ) THEN
       REWIND( UNIT = buffer )
     ELSE
       OPEN( UNIT = buffer )
     END IF

!  copy the contents of ARRAY into the buffer i/o area

     WRITE( UNIT = buffer, FMT = * ) ARRAY( 0 : used_length - 1 )

!  extend the length of ARRAY

     DEALLOCATE( ARRAY )

 110 CONTINUE
     ALLOCATE( ARRAY( 0 : new_length - 1 ), STAT = alloc_status )

!  if the allocation failed, reduce the new length and retry

     IF ( alloc_status /= 0 ) THEN
       new_length = min_length + ( new_length - min_length ) / 2
       IF ( new_length < min_length ) THEN
          status = GALAHAD_error_allocate
          RETURN
       END IF
       GO TO 110
     END IF

!  copy the contents of ARRAY back from the buffer i/o area

     REWIND( UNIT = buffer )
     READ( UNIT = buffer, FMT = * ) ARRAY( 0 : used_length - 1 )

!  successful exit

 200 CONTINUE
     status = GALAHAD_ok
     RETURN

!  end of subroutine SPACE_extend_carray_real

     END SUBROUTINE SPACE_extend_carray_real

!-  S P A C E _ D E A L L O C _ R E A L _ P O I N T E R   S U B R O U T I N E  -

     SUBROUTINE SPACE_dealloc_real_pointer( point, status, alloc_status,       &
                                            point_name, bad_alloc, out )

!  Deallocate the real pointer array "point"

!  Dummy arguments

     INTEGER ( KIND = ip_ ), INTENT( OUT ) :: status, alloc_status
     REAL ( KIND = rp_ ), POINTER, DIMENSION( : ) :: point
     INTEGER ( KIND = ip_ ), OPTIONAL :: out
     CHARACTER ( LEN = 80 ), OPTIONAL :: point_name
     CHARACTER ( LEN = 80 ), OPTIONAL :: bad_alloc

     status = GALAHAD_ok ; alloc_status = 0
     IF ( PRESENT( bad_alloc ) ) bad_alloc = ''
     IF ( ASSOCIATED( point ) ) THEN
       DEALLOCATE( point, STAT = alloc_status )
       IF ( alloc_status /= 0 ) THEN
         status = GALAHAD_error_deallocate
         IF ( PRESENT( bad_alloc ) .AND. PRESENT( point_name ) )               &
           bad_alloc = point_name
         IF ( PRESENT( out ) ) THEN
           IF ( PRESENT( point_name ) ) THEN
             IF ( out > 0 ) WRITE( out, 2900 ) TRIM( point_name ), alloc_status
           ELSE
             IF ( out > 0 ) WRITE( out, 2910 ) alloc_status
           END IF
         END IF
       END IF
     END IF
     RETURN

!  Non-executable statements

2900 FORMAT( ' ** Deallocation error for ', A, /, '     status = ', I6 )
2910 FORMAT( ' ** Deallocation error status = ', I6 )

!  End of subroutine SPACE_dealloc_real_pointer

     END SUBROUTINE SPACE_dealloc_real_pointer

!-  S P A C E _ D E A L L O C _ R E A L 2 _ P O I N T E R  S U B R O U T I N E -

     SUBROUTINE SPACE_dealloc_real2_pointer( point, status, alloc_status,      &
                                             point_name, bad_alloc, out )

!  Deallocate the rank-2 real pointer array "point"

!  Dummy arguments

     INTEGER ( KIND = ip_ ), INTENT( OUT ) :: status, alloc_status
     REAL ( KIND = rp_ ), POINTER, DIMENSION( : , : ) :: point
     INTEGER ( KIND = ip_ ), OPTIONAL :: out
     CHARACTER ( LEN = 80 ), OPTIONAL :: point_name
     CHARACTER ( LEN = 80 ), OPTIONAL :: bad_alloc

     status = GALAHAD_ok ; alloc_status = 0
     IF ( PRESENT( bad_alloc ) ) bad_alloc = ''
     IF ( ASSOCIATED( point ) ) THEN
       DEALLOCATE( point, STAT = alloc_status )
       IF ( alloc_status /= 0 ) THEN
         status = GALAHAD_error_deallocate
         IF ( PRESENT( bad_alloc ) .AND. PRESENT( point_name ) )               &
           bad_alloc = point_name
         IF ( PRESENT( out ) ) THEN
           IF ( PRESENT( point_name ) ) THEN
             IF ( out > 0 ) WRITE( out, 2900 ) TRIM( point_name ), alloc_status
           ELSE
             IF ( out > 0 ) WRITE( out, 2910 ) alloc_status
           END IF
         END IF
       END IF
     END IF
     RETURN

!  Non-executable statements

2900 FORMAT( ' ** Deallocation error for ', A, /, '     status = ', I6 )
2910 FORMAT( ' ** Deallocation error status = ', I6 )

!  End of subroutine SPACE_dealloc_real2_pointer

     END SUBROUTINE SPACE_dealloc_real2_pointer

!-  S P A C E _ D E A L L O C _ I N T E G E R _ P O I N T E R   SUBROUTINE  -*-

     SUBROUTINE SPACE_dealloc_integer_pointer( point, status, alloc_status,    &
                                               point_name, bad_alloc, out )

!  Deallocate the integer pointer array "point"

!  Dummy arguments

     INTEGER ( KIND = ip_ ), INTENT( OUT ) :: status, alloc_status
     INTEGER ( KIND = ip_ ), POINTER, DIMENSION( : ) :: point
     INTEGER ( KIND = ip_ ), OPTIONAL :: out
     CHARACTER ( LEN = 80 ), OPTIONAL :: point_name
     CHARACTER ( LEN = 80 ), OPTIONAL :: bad_alloc

     status = GALAHAD_ok ; alloc_status = 0
     IF ( PRESENT( bad_alloc ) ) bad_alloc = ''
     IF ( ASSOCIATED( point ) ) THEN
       DEALLOCATE( point, STAT = alloc_status )
       IF ( alloc_status /= 0 ) THEN
         status = GALAHAD_error_deallocate
         IF ( PRESENT( bad_alloc ) .AND. PRESENT( point_name ) )               &
           bad_alloc = point_name
         IF ( PRESENT( out ) ) THEN
           IF ( PRESENT( point_name ) ) THEN
             IF ( out > 0 ) WRITE( out, 2900 ) TRIM( point_name ), alloc_status
           ELSE
             IF ( out > 0 ) WRITE( out, 2910 ) alloc_status
           END IF
         END IF
       END IF
     END IF
     RETURN

!  Non-executable statements

2900 FORMAT( ' ** Deallocation error for ', A, /, '     status = ', I6 )
2910 FORMAT( ' ** Deallocation error status = ', I6 )

!  End of subroutine SPACE_dealloc_integer_pointer

     END SUBROUTINE SPACE_dealloc_integer_pointer

! -*- S P A C E _ D E A L L O C _ I N T E G E R 2 _ P O I N T E R  SUBROUTINE -*

     SUBROUTINE SPACE_dealloc_integer2_pointer( point, status, alloc_status,   &
                                                point_name, bad_alloc, out )

!  Deallocate the rank-2 integer pointer array "point"

!  Dummy arguments

     INTEGER ( KIND = ip_ ), INTENT( OUT ) :: status, alloc_status
     INTEGER ( KIND = ip_ ), POINTER, DIMENSION( : , : ) :: point
     INTEGER ( KIND = ip_ ), OPTIONAL :: out
     CHARACTER ( LEN = 80 ), OPTIONAL :: point_name
     CHARACTER ( LEN = 80 ), OPTIONAL :: bad_alloc

     status = GALAHAD_ok ; alloc_status = 0
     IF ( PRESENT( bad_alloc ) ) bad_alloc = ''
     IF ( ASSOCIATED( point ) ) THEN
       DEALLOCATE( point, STAT = alloc_status )
       IF ( alloc_status /= 0 ) THEN
         status = GALAHAD_error_deallocate
         IF ( PRESENT( bad_alloc ) .AND. PRESENT( point_name ) )               &
           bad_alloc = point_name
         IF ( PRESENT( out ) ) THEN
           IF ( PRESENT( point_name ) ) THEN
             IF ( out > 0 ) WRITE( out, 2900 ) TRIM( point_name ), alloc_status
           ELSE
             IF ( out > 0 ) WRITE( out, 2910 ) alloc_status
           END IF
         END IF
       END IF
     END IF
     RETURN

!  Non-executable statements

2900 FORMAT( ' ** Deallocation error for ', A, /, '     status = ', I6 )
2910 FORMAT( ' ** Deallocation error status = ', I6 )

!  End of subroutine SPACE_dealloc_integer2_pointer

     END SUBROUTINE SPACE_dealloc_integer2_pointer

!-*-  S P A C E _ D E A L L O C _ L O G I C A L _ P O I N T E R   SUBROUTINE -*-

     SUBROUTINE SPACE_dealloc_logical_pointer( point, status, alloc_status,    &
                                               point_name, bad_alloc, out )

!  Deallocate the logical pointer array "point"

!  Dummy arguments

     INTEGER ( KIND = ip_ ), INTENT( OUT ) :: status, alloc_status
     LOGICAL ( KIND = i4_ ), POINTER, DIMENSION( : ) :: point
     INTEGER ( KIND = ip_ ), OPTIONAL :: out
     CHARACTER ( LEN = 80 ), OPTIONAL :: point_name
     CHARACTER ( LEN = 80 ), OPTIONAL :: bad_alloc

     status = GALAHAD_ok ; alloc_status = 0
     IF ( PRESENT( bad_alloc ) ) bad_alloc = ''
     IF ( ASSOCIATED( point ) ) THEN
       DEALLOCATE( point, STAT = alloc_status )
       IF ( alloc_status /= 0 ) THEN
         status = GALAHAD_error_deallocate
         IF ( PRESENT( bad_alloc ) .AND. PRESENT( point_name ) )               &
           bad_alloc = point_name
         IF ( PRESENT( out ) ) THEN
           IF ( PRESENT( point_name ) ) THEN
             IF ( out > 0 ) WRITE( out, 2900 ) TRIM( point_name ), alloc_status
           ELSE
             IF ( out > 0 ) WRITE( out, 2910 ) alloc_status
           END IF
         END IF
       END IF
     END IF
     RETURN

!  Non-executable statements

2900 FORMAT( ' ** Deallocation error for ', A, /, '     status = ', I6 )
2910 FORMAT( ' ** Deallocation error status = ', I6 )

!  End of subroutine SPACE_dealloc_logical_pointer

     END SUBROUTINE SPACE_dealloc_logical_pointer

!-  S P A C E _ D E A L L O C _ L O G I C A L 6 4 _ P O I N T E R   SUBROUTINE -

     SUBROUTINE SPACE_dealloc_logical64_pointer( point, status, alloc_status,  &
                                                 point_name, bad_alloc, out )

!  Deallocate the logical pointer array "point"

!  Dummy arguments

     INTEGER ( KIND = ip_ ), INTENT( OUT ) :: status, alloc_status
     LOGICAL ( KIND = i8_ ), POINTER, DIMENSION( : ) :: point
     INTEGER ( KIND = ip_ ), OPTIONAL :: out
     CHARACTER ( LEN = 80 ), OPTIONAL :: point_name
     CHARACTER ( LEN = 80 ), OPTIONAL :: bad_alloc

     status = GALAHAD_ok ; alloc_status = 0
     IF ( PRESENT( bad_alloc ) ) bad_alloc = ''
     IF ( ASSOCIATED( point ) ) THEN
       DEALLOCATE( point, STAT = alloc_status )
       IF ( alloc_status /= 0 ) THEN
         status = GALAHAD_error_deallocate
         IF ( PRESENT( bad_alloc ) .AND. PRESENT( point_name ) )               &
           bad_alloc = point_name
         IF ( PRESENT( out ) ) THEN
           IF ( PRESENT( point_name ) ) THEN
             IF ( out > 0 ) WRITE( out, 2900 ) TRIM( point_name ), alloc_status
           ELSE
             IF ( out > 0 ) WRITE( out, 2910 ) alloc_status
           END IF
         END IF
       END IF
     END IF
     RETURN

!  Non-executable statements

2900 FORMAT( ' ** Deallocation error for ', A, /, '     status = ', I6 )
2910 FORMAT( ' ** Deallocation error status = ', I6 )

!  End of subroutine SPACE_dealloc_logical64_pointer

     END SUBROUTINE SPACE_dealloc_logical64_pointer

!-  S P A C E _ D E A L L O C _ C H A R A C T E R  _ P O I N T E R  SUBROUTINE -

     SUBROUTINE SPACE_dealloc_character_pointer( point, status, alloc_status,  &
                                                 point_name, bad_alloc, out )

!  Deallocate the character pointer array "point"

!  Dummy arguments

     INTEGER ( KIND = ip_ ), INTENT( OUT ) :: status, alloc_status
     CHARACTER( LEN = * ), POINTER, DIMENSION( : ) :: point
     INTEGER ( KIND = ip_ ), OPTIONAL :: out
     CHARACTER ( LEN = 80 ), OPTIONAL :: point_name
     CHARACTER ( LEN = 80 ), OPTIONAL :: bad_alloc

     status = GALAHAD_ok ; alloc_status = 0
     IF ( PRESENT( bad_alloc ) ) bad_alloc = ''
     IF ( ASSOCIATED( point ) ) THEN
       DEALLOCATE( point, STAT = alloc_status )
       IF ( alloc_status /= 0 ) THEN
         status = GALAHAD_error_deallocate
         IF ( PRESENT( bad_alloc ) .AND. PRESENT( point_name ) )               &
           bad_alloc = point_name
         IF ( PRESENT( out ) ) THEN
           IF ( PRESENT( point_name ) ) THEN
             IF ( out > 0 ) WRITE( out, 2900 ) TRIM( point_name ), alloc_status
           ELSE
             IF ( out > 0 ) WRITE( out, 2910 ) alloc_status
           END IF
         END IF
       END IF
     END IF
     RETURN

!  Non-executable statements

2900 FORMAT( ' ** Deallocation error for ', A, /, '     status = ', I6 )
2910 FORMAT( ' ** Deallocation error status = ', I6 )

!  End of subroutine SPACE_dealloc_character_pointer

     END SUBROUTINE SPACE_dealloc_character_pointer

!-  S P A C E _ D E A L L O C _ C H A R A C T E R 2 _ P O I N T E R  SUBROUTINE

     SUBROUTINE SPACE_dealloc_character2_pointer( point, status, alloc_status, &
                                                  point_name, bad_alloc, out )

!  Deallocate the character pointer array "point"

!  Dummy arguments

     INTEGER ( KIND = ip_ ), INTENT( OUT ) :: status, alloc_status
     CHARACTER( LEN = * ), POINTER, DIMENSION( : , : ) :: point
     INTEGER ( KIND = ip_ ), OPTIONAL :: out
     CHARACTER ( LEN = 80 ), OPTIONAL :: point_name
     CHARACTER ( LEN = 80 ), OPTIONAL :: bad_alloc

     status = GALAHAD_ok ; alloc_status = 0
     IF ( PRESENT( bad_alloc ) ) bad_alloc = ''
     IF ( ASSOCIATED( point ) ) THEN
       DEALLOCATE( point, STAT = alloc_status )
       IF ( alloc_status /= 0 ) THEN
         status = GALAHAD_error_deallocate
         IF ( PRESENT( bad_alloc ) .AND. PRESENT( point_name ) )               &
           bad_alloc = point_name
         IF ( PRESENT( out ) ) THEN
           IF ( PRESENT( point_name ) ) THEN
             IF ( out > 0 ) WRITE( out, 2900 ) TRIM( point_name ), alloc_status
           ELSE
             IF ( out > 0 ) WRITE( out, 2910 ) alloc_status
           END IF
         END IF
       END IF
     END IF
     RETURN

!  Non-executable statements

2900 FORMAT( ' ** Deallocation error for ', A, /, '     status = ', I6 )
2910 FORMAT( ' ** Deallocation error status = ', I6 )

!  End of subroutine SPACE_dealloc_character2_pointer

     END SUBROUTINE SPACE_dealloc_character2_pointer

!-*-  S P A C E _ D E A L L O C _ R E A L _ A R R A Y   S U B R O U T I N E  -*-

     SUBROUTINE SPACE_dealloc_real_array( array, status, alloc_status,         &
                                          array_name, bad_alloc, out )

!  Deallocate the real allocatable array "array"

!  Dummy arguments

     INTEGER ( KIND = ip_ ), INTENT( OUT ) :: status, alloc_status
     REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : ) :: array
     INTEGER ( KIND = ip_ ), OPTIONAL :: out
     CHARACTER ( LEN = 80 ), OPTIONAL :: array_name
     CHARACTER ( LEN = 80 ), OPTIONAL :: bad_alloc

     status = GALAHAD_ok ; alloc_status = 0
     IF ( PRESENT( bad_alloc ) ) bad_alloc = ''
     IF ( ALLOCATED( array) ) THEN
       DEALLOCATE( array, STAT = alloc_status )
       IF ( alloc_status /= 0 ) THEN
         status = GALAHAD_error_deallocate
         IF ( PRESENT( bad_alloc ) .AND. PRESENT( array_name ) )              &
           bad_alloc = array_name
         IF ( PRESENT( out ) ) THEN
           IF ( PRESENT( array_name ) ) THEN
             IF ( out > 0 ) WRITE( out, 2900 ) TRIM( array_name ), alloc_status
           ELSE
             IF ( out > 0 ) WRITE( out, 2910 ) alloc_status
           END IF
         END IF
       END IF
     END IF
     RETURN

!  Non-executable statements

2900 FORMAT( ' ** Deallocation error for ', A, /, '     status = ', I6 )
2910 FORMAT( ' ** Deallocation error status = ', I6 )

!  End of subroutine SPACE_dealloc_real_array

     END SUBROUTINE SPACE_dealloc_real_array

!-*-  S P A C E _ D E A L L O C _ R E A L 2 _ A R R A Y  S U B R O U T I N E -*-

     SUBROUTINE SPACE_dealloc_real2_array( array, status, alloc_status,        &
                                           array_name, bad_alloc, out )

!  Deallocate the rank-2 real allocatable array "array"

!  Dummy arguments

     INTEGER ( KIND = ip_ ), INTENT( OUT ) :: status, alloc_status
     REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : , : ) :: array
     INTEGER ( KIND = ip_ ), OPTIONAL :: out
     CHARACTER ( LEN = 80 ), OPTIONAL :: array_name
     CHARACTER ( LEN = 80 ), OPTIONAL :: bad_alloc

     status = GALAHAD_ok ; alloc_status = 0
     IF ( PRESENT( bad_alloc ) ) bad_alloc = ''
     IF ( ALLOCATED( array) ) THEN
       DEALLOCATE( array, STAT = alloc_status )
       IF ( alloc_status /= 0 ) THEN
         status = GALAHAD_error_deallocate
         IF ( PRESENT( bad_alloc ) .AND. PRESENT( array_name ) )               &
           bad_alloc = array_name
         IF ( PRESENT( out ) ) THEN
           IF ( PRESENT( array_name ) ) THEN
             IF ( out > 0 ) WRITE( out, 2900 ) TRIM( array_name ), alloc_status
           ELSE
             IF ( out > 0 ) WRITE( out, 2910 ) alloc_status
           END IF
         END IF
       END IF
     END IF
     RETURN

!  Non-executable statements

2900 FORMAT( ' ** Deallocation error for ', A, /, '     status = ', I6 )
2910 FORMAT( ' ** Deallocation error status = ', I6 )

!  End of subroutine SPACE_dealloc_real2_array

     END SUBROUTINE SPACE_dealloc_real2_array

!-*-  S P A C E _ D E A L L O C _ R E A L 3 _ A R R A Y  S U B R O U T I N E -*-

     SUBROUTINE SPACE_dealloc_real3_array( array, status, alloc_status,        &
                                           array_name, bad_alloc, out )

!  Deallocate the rank-2 real allocatable array "array"

!  Dummy arguments

     INTEGER ( KIND = ip_ ), INTENT( OUT ) :: status, alloc_status
     REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : , : , : ) :: array
     INTEGER ( KIND = ip_ ), OPTIONAL :: out
     CHARACTER ( LEN = 80 ), OPTIONAL :: array_name
     CHARACTER ( LEN = 80 ), OPTIONAL :: bad_alloc

     status = GALAHAD_ok ; alloc_status = 0
     IF ( PRESENT( bad_alloc ) ) bad_alloc = ''
     IF ( ALLOCATED( array) ) THEN
       DEALLOCATE( array, STAT = alloc_status )
       IF ( alloc_status /= 0 ) THEN
         status = GALAHAD_error_deallocate
         IF ( PRESENT( bad_alloc ) .AND. PRESENT( array_name ) )               &
           bad_alloc = array_name
         IF ( PRESENT( out ) ) THEN
           IF ( PRESENT( array_name ) ) THEN
             IF ( out > 0 ) WRITE( out, 2900 ) TRIM( array_name ), alloc_status
           ELSE
             IF ( out > 0 ) WRITE( out, 2910 ) alloc_status
           END IF
         END IF
       END IF
     END IF
     RETURN

!  Non-executable statements

2900 FORMAT( ' ** Deallocation error for ', A, /, '     status = ', I6 )
2910 FORMAT( ' ** Deallocation error status = ', I6 )

!  End of subroutine SPACE_dealloc_real3_array

     END SUBROUTINE SPACE_dealloc_real3_array

!- S P A C E _ D E A L L O C _ C O M P L E X _ A R R A Y   S U B R O U T I N E -

     SUBROUTINE SPACE_dealloc_complex_array( array, status, alloc_status,      &
                                             array_name, bad_alloc, out )

!  Deallocate the complex allocatable array "array"

!  Dummy arguments

     INTEGER ( KIND = ip_ ), INTENT( OUT ) :: status, alloc_status
     COMPLEX ( KIND = cp_ ), ALLOCATABLE, DIMENSION( : ) :: array
     INTEGER ( KIND = ip_ ), OPTIONAL :: out
     CHARACTER ( LEN = 80 ), OPTIONAL :: array_name
     CHARACTER ( LEN = 80 ), OPTIONAL :: bad_alloc

     status = GALAHAD_ok ; alloc_status = 0
     IF ( PRESENT( bad_alloc ) ) bad_alloc = ''
     IF ( ALLOCATED( array ) ) THEN
       DEALLOCATE( array, STAT = alloc_status )
       IF ( alloc_status /= 0 ) THEN
         status = GALAHAD_error_deallocate
         IF ( PRESENT( bad_alloc ) .AND. PRESENT( array_name ) )               &
           bad_alloc = array_name
         IF ( PRESENT( out ) ) THEN
           IF ( PRESENT( array_name ) ) THEN
             IF ( out > 0 ) WRITE( out, 2900 ) TRIM( array_name ), alloc_status
           ELSE
             IF ( out > 0 ) WRITE( out, 2910 ) alloc_status
           END IF
         END IF
       END IF
     END IF
     RETURN

!  Non-executable statements

2900 FORMAT( ' ** Deallocation error for ', A, /, '     status = ', I6 )
2910 FORMAT( ' ** Deallocation error status = ', I6 )

!  End of subroutine SPACE_dealloc_complex_array

     END SUBROUTINE SPACE_dealloc_complex_array

!- S P A C E _ D E A L L O C _ I N T E G E R _ A R R A Y   S U B R O U T I N E -

     SUBROUTINE SPACE_dealloc_integer_array( array, status, alloc_status,      &
                                             array_name, bad_alloc, out )

!  Deallocate the integer allocatable array "array"

!  Dummy arguments

     INTEGER ( KIND = ip_ ), INTENT( OUT ) :: status, alloc_status
     INTEGER ( KIND = ip_ ), ALLOCATABLE, DIMENSION( : ) :: array
     INTEGER ( KIND = ip_ ), OPTIONAL :: out
     CHARACTER ( LEN = 80 ), OPTIONAL :: array_name
     CHARACTER ( LEN = 80 ), OPTIONAL :: bad_alloc

     status = GALAHAD_ok ; alloc_status = 0
     IF ( PRESENT( bad_alloc ) ) bad_alloc = ''
     IF ( ALLOCATED( array) ) THEN
       DEALLOCATE( array, STAT = alloc_status )
       IF ( alloc_status /= 0 ) THEN
         status = GALAHAD_error_deallocate
         IF ( PRESENT( bad_alloc ) .AND. PRESENT( array_name ) )              &
           bad_alloc = array_name
         IF ( PRESENT( out ) ) THEN
           IF ( PRESENT( array_name ) ) THEN
             IF ( out > 0 ) WRITE( out, 2900 ) TRIM( array_name ), alloc_status
           ELSE
             IF ( out > 0 ) WRITE( out, 2910 ) alloc_status
           END IF
         END IF
       END IF
     END IF
     RETURN

!  Non-executable statements

2900 FORMAT( ' ** Deallocation error for ', A, /, '     status = ', I6 )
2910 FORMAT( ' ** Deallocation error status = ', I6 )

!  End of subroutine SPACE_dealloc_integer_array

     END SUBROUTINE SPACE_dealloc_integer_array

!- S P A C E _ D E A L L O C _ I N T E G E R 2 _ A R R A Y  S U B R O U T I N E

     SUBROUTINE SPACE_dealloc_integer2_array( array, status, alloc_status,     &
                                              array_name, bad_alloc, out )

!  Deallocate the rank-2 integer allocatable array "array"

!  Dummy arguments

     INTEGER ( KIND = ip_ ), INTENT( OUT ) :: status, alloc_status
     INTEGER ( KIND = ip_ ), ALLOCATABLE, DIMENSION( : , : ) :: array
     INTEGER ( KIND = ip_ ), OPTIONAL :: out
     CHARACTER ( LEN = 80 ), OPTIONAL :: array_name
     CHARACTER ( LEN = 80 ), OPTIONAL :: bad_alloc

     status = GALAHAD_ok ; alloc_status = 0
     IF ( PRESENT( bad_alloc ) ) bad_alloc = ''
     IF ( ALLOCATED( array) ) THEN
       DEALLOCATE( array, STAT = alloc_status )
       IF ( alloc_status /= 0 ) THEN
         status = GALAHAD_error_deallocate
         IF ( PRESENT( bad_alloc ) .AND. PRESENT( array_name ) )               &
           bad_alloc = array_name
         IF ( PRESENT( out ) ) THEN
           IF ( PRESENT( array_name ) ) THEN
             IF ( out > 0 ) WRITE( out, 2900 ) TRIM( array_name ), alloc_status
           ELSE
             IF ( out > 0 ) WRITE( out, 2910 ) alloc_status
           END IF
         END IF
       END IF
     END IF
     RETURN

!  Non-executable statements

2900 FORMAT( ' ** Deallocation error for ', A, /, '     status = ', I6 )
2910 FORMAT( ' ** Deallocation error status = ', I6 )

!  End of subroutine SPACE_dealloc_integer2_array

     END SUBROUTINE SPACE_dealloc_integer2_array

!-*-*-  S P A C E _ D E A L L O C _ L O G I C A L _ A R R A Y   SUBROUTINE -*-*-

     SUBROUTINE SPACE_dealloc_logical_array( array, status, alloc_status,      &
                                             array_name, bad_alloc, out )

!  Deallocate the logical allocatable array "array"

!  Dummy arguments

     INTEGER ( KIND = ip_ ), INTENT( OUT ) :: status, alloc_status
     LOGICAL ( KIND = i4_ ), ALLOCATABLE, DIMENSION( : ) :: array
     INTEGER ( KIND = ip_ ), OPTIONAL :: out
     CHARACTER ( LEN = 80 ), OPTIONAL :: array_name
     CHARACTER ( LEN = 80 ), OPTIONAL :: bad_alloc

     status = GALAHAD_ok ; alloc_status = 0
     IF ( PRESENT( bad_alloc ) ) bad_alloc = ''
     IF ( ALLOCATED( array) ) THEN
       DEALLOCATE( array, STAT = alloc_status )
       IF ( alloc_status /= 0 ) THEN
         status = GALAHAD_error_deallocate
         IF ( PRESENT( bad_alloc ) .AND. PRESENT( array_name ) )               &
           bad_alloc = array_name
         IF ( PRESENT( out ) ) THEN
           IF ( PRESENT( array_name ) ) THEN
             IF ( out > 0 ) WRITE( out, 2900 ) TRIM( array_name ), alloc_status
           ELSE
             IF ( out > 0 ) WRITE( out, 2910 ) alloc_status
           END IF
         END IF
       END IF
     END IF
     RETURN

!  Non-executable statements

2900 FORMAT( ' ** Deallocation error for ', A, /, '     status = ', I6 )
2910 FORMAT( ' ** Deallocation error status = ', I6 )

!  End of subroutine SPACE_dealloc_logical_array

     END SUBROUTINE SPACE_dealloc_logical_array

!-*-  S P A C E _ D E A L L O C _ L O G I C A L 6 4 _ A R R A Y   SUBROUTINE -*-

     SUBROUTINE SPACE_dealloc_logical64_array( array, status, alloc_status,    &
                                               array_name, bad_alloc, out )

!  Deallocate the logical allocatable array "array"

!  Dummy arguments

     INTEGER ( KIND = ip_ ), INTENT( OUT ) :: status, alloc_status
     LOGICAL ( KIND = i8_ ), ALLOCATABLE, DIMENSION( : ) :: array
     INTEGER ( KIND = ip_ ), OPTIONAL :: out
     CHARACTER ( LEN = 80 ), OPTIONAL :: array_name
     CHARACTER ( LEN = 80 ), OPTIONAL :: bad_alloc

     status = GALAHAD_ok ; alloc_status = 0
     IF ( PRESENT( bad_alloc ) ) bad_alloc = ''
     IF ( ALLOCATED( array) ) THEN
       DEALLOCATE( array, STAT = alloc_status )
       IF ( alloc_status /= 0 ) THEN
         status = GALAHAD_error_deallocate
         IF ( PRESENT( bad_alloc ) .AND. PRESENT( array_name ) )               &
           bad_alloc = array_name
         IF ( PRESENT( out ) ) THEN
           IF ( PRESENT( array_name ) ) THEN
             IF ( out > 0 ) WRITE( out, 2900 ) TRIM( array_name ), alloc_status
           ELSE
             IF ( out > 0 ) WRITE( out, 2910 ) alloc_status
           END IF
         END IF
       END IF
     END IF
     RETURN

!  Non-executable statements

2900 FORMAT( ' ** Deallocation error for ', A, /, '     status = ', I6 )
2910 FORMAT( ' ** Deallocation error status = ', I6 )

!  End of subroutine SPACE_dealloc_logical64_array

     END SUBROUTINE SPACE_dealloc_logical64_array

!-  S P A C E _ D E A L L O C _ C H A R A C T E R  _ A R R A Y  SUBROUTINE  -*

     SUBROUTINE SPACE_dealloc_character_array( array, status, alloc_status,    &
                                               array_name, bad_alloc, out )

!  Deallocate the character allocatable array "array"

!  Dummy arguments

     INTEGER ( KIND = ip_ ), INTENT( OUT ) :: status, alloc_status
     CHARACTER( LEN = * ), ALLOCATABLE, DIMENSION( : ) :: array
     INTEGER ( KIND = ip_ ), OPTIONAL :: out
     CHARACTER ( LEN = 80 ), OPTIONAL :: array_name
     CHARACTER ( LEN = 80 ), OPTIONAL :: bad_alloc

     status = GALAHAD_ok ; alloc_status = 0
     IF ( PRESENT( bad_alloc ) ) bad_alloc = ''
     IF ( ALLOCATED( array) ) THEN
       DEALLOCATE( array, STAT = alloc_status )
       IF ( alloc_status /= 0 ) THEN
         status = GALAHAD_error_deallocate
         IF ( PRESENT( bad_alloc ) .AND. PRESENT( array_name ) )               &
           bad_alloc = array_name
         IF ( PRESENT( out ) ) THEN
           IF ( PRESENT( array_name ) ) THEN
             IF ( out > 0 ) WRITE( out, 2900 ) TRIM( array_name ), alloc_status
           ELSE
             IF ( out > 0 ) WRITE( out, 2910 ) alloc_status
           END IF
         END IF
       END IF
     END IF
     RETURN

!  Non-executable statements

2900 FORMAT( ' ** Deallocation error for ', A, /, '     status = ', I6 )
2910 FORMAT( ' ** Deallocation error status = ', I6 )

!  End of subroutine SPACE_dealloc_character_array

     END SUBROUTINE SPACE_dealloc_character_array

!-  S P A C E _ D E A L L O C _ C H A R A C T E R 2 _ A R R A Y  SUBROUTINE  -*

     SUBROUTINE SPACE_dealloc_character2_array( array, status, alloc_status,   &
                                                array_name, bad_alloc, out )

!  Deallocate the character allocatable array "array"

!  Dummy arguments

     INTEGER ( KIND = ip_ ), INTENT( OUT ) :: status, alloc_status
     CHARACTER( LEN = * ), ALLOCATABLE, DIMENSION( : , : ) :: array
     INTEGER ( KIND = ip_ ), OPTIONAL :: out
     CHARACTER ( LEN = 80 ), OPTIONAL :: array_name
     CHARACTER ( LEN = 80 ), OPTIONAL :: bad_alloc

     status = GALAHAD_ok ; alloc_status = 0
     IF ( PRESENT( bad_alloc ) ) bad_alloc = ''
     IF ( ALLOCATED( array) ) THEN
       DEALLOCATE( array, STAT = alloc_status )
       IF ( alloc_status /= 0 ) THEN
         status = GALAHAD_error_deallocate
         IF ( PRESENT( bad_alloc ) .AND. PRESENT( array_name ) )               &
           bad_alloc = array_name
         IF ( PRESENT( out ) ) THEN
           IF ( PRESENT( array_name ) ) THEN
             IF ( out > 0 ) WRITE( out, 2900 ) TRIM( array_name ), alloc_status
           ELSE
             IF ( out > 0 ) WRITE( out, 2910 ) alloc_status
           END IF
         END IF
       END IF
     END IF
     RETURN

!  Non-executable statements

2900 FORMAT( ' ** Deallocation error for ', A, /, '     status = ', I6 )
2910 FORMAT( ' ** Deallocation error status = ', I6 )

!  End of subroutine SPACE_dealloc_character2_array

     END SUBROUTINE SPACE_dealloc_character2_array

!-*-*-  S P A C E _ D E A L L O C _ S M T _ T Y P E   S U B R O U T I N E  -*-*-

     SUBROUTINE SPACE_dealloc_smt_type( array, status, alloc_status,           &
                                        array_name, bad_alloc, out )

!  Deallocate the components of a variable of type SMT_type

!  Dummy arguments

     INTEGER ( KIND = ip_ ), INTENT( OUT ) :: status, alloc_status
     TYPE ( SMT_type ) :: array
     INTEGER ( KIND = ip_ ), OPTIONAL :: out
     CHARACTER ( LEN = 80 ), OPTIONAL :: array_name
     CHARACTER ( LEN = 80 ), OPTIONAL :: bad_alloc

!  Local variables

     INTEGER ( KIND = ip_ ) :: status_comp, alloc_status_comp

     status = GALAHAD_ok ; alloc_status = 0
     IF ( PRESENT( bad_alloc ) ) bad_alloc = ''

     CALL SPACE_dealloc_array( array%id, status_comp, alloc_status_comp,       &
                               array_name, bad_alloc, out )
     IF ( alloc_status_comp /= 0 ) THEN
       status = GALAHAD_error_deallocate
       alloc_status = alloc_status_comp
       IF ( PRESENT( bad_alloc ) .AND. PRESENT( array_name ) )                 &
         bad_alloc = array_name
     END IF

     CALL SPACE_dealloc_array( array%type, status_comp, alloc_status_comp,     &
                               array_name, bad_alloc, out )
     IF ( alloc_status_comp /= 0 ) THEN
       status = GALAHAD_error_deallocate
       alloc_status = alloc_status_comp
       IF ( PRESENT( bad_alloc ) .AND. PRESENT( array_name ) )                 &
         bad_alloc = array_name
     END IF

     CALL SPACE_dealloc_array( array%row, status_comp, alloc_status_comp,      &
                               array_name, bad_alloc, out )
     IF ( alloc_status_comp /= 0 ) THEN
       status = GALAHAD_error_deallocate
       alloc_status = alloc_status_comp
       IF ( PRESENT( bad_alloc ) .AND. PRESENT( array_name ) )                 &
         bad_alloc = array_name
     END IF

     CALL SPACE_dealloc_array( array%col, status_comp, alloc_status_comp,      &
                               array_name, bad_alloc, out )
     IF ( alloc_status_comp /= 0 ) THEN
       status = GALAHAD_error_deallocate
       alloc_status = alloc_status_comp
       IF ( PRESENT( bad_alloc ) .AND. PRESENT( array_name ) )                 &
         bad_alloc = array_name
     END IF

     CALL SPACE_dealloc_array( array%ptr, status_comp, alloc_status_comp,      &
                               array_name, bad_alloc, out )
     IF ( alloc_status_comp /= 0 ) THEN
       status = GALAHAD_error_deallocate
       alloc_status = alloc_status_comp
       IF ( PRESENT( bad_alloc ) .AND. PRESENT( array_name ) )                 &
         bad_alloc = array_name
     END IF

!    CALL SPACE_dealloc_array( array%val, status_comp, alloc_status_comp,      &
     CALL SPACE_dealloc_real_array( array%val, status_comp, alloc_status_comp,      &
                               array_name, bad_alloc, out )
     IF ( alloc_status_comp /= 0 ) THEN
       status = GALAHAD_error_deallocate
       alloc_status = alloc_status_comp
       IF ( PRESENT( bad_alloc ) .AND. PRESENT( array_name ) )                 &
         bad_alloc = array_name
     END IF

     RETURN

!  End of subroutine SPACE_dealloc_smt_type

     END SUBROUTINE SPACE_dealloc_smt_type

!  End of module GALAHAD_SPACE

   END MODULE GALAHAD_SPACE_precision
