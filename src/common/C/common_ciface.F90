! THIS VERSION: GALAHAD 4.1 - 2023-01-24 AT 09:30 GMT.

#include "galahad_modules.h"
#include "galahad_cfunctions.h"

!-*-*-*-*-*-*-  G A L A H A D _ C O M M O N   C   I N T E R F A C E  -*-*-*-*-

!  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
!  Principal authors: Jaroslav Fowkes & Nick Gould

!  History -
!    originally released GALAHAD Version 3.3. July 25rd 2021

!  For full documentation, see
!   http://galahad.rl.ac.uk/galahad-www/specs.html

!  C interface module for common GALAHAD interfaces

  MODULE GALAHAD_common_ciface

    USE GALAHAD_KINDS_precision

    IMPLICIT NONE
    PUBLIC

!----------------------
!   I n t e r f a c e s
!----------------------

    INTERFACE
      INTEGER ( KIND = C_SIZE_T ) PURE FUNCTION strlen( cstr ) BIND( C )
        USE GALAHAD_KINDS_precision
        IMPLICIT NONE
        TYPE ( C_PTR ), INTENT( IN ), VALUE :: cstr
      END FUNCTION strlen
    END INTERFACE

!----------------------
!   P r o c e d u r e s
!----------------------

  CONTAINS

!  optional string length

    PURE FUNCTION opt_strlen( cstr ) RESULT( len )
    TYPE ( C_PTR ), INTENT( IN ), VALUE :: cstr
    INTEGER ( KIND = C_SIZE_T ) :: len

    IF ( C_ASSOCIATED( cstr ) ) THEN
      len = strlen( cstr )
    ELSE
      len = 0
    END IF
    RETURN

    END FUNCTION opt_strlen

!  copy C string to fortran character string

    FUNCTION cstr_to_fchar( cstr ) RESULT( fchar )
    TYPE ( C_PTR ) :: cstr
    CHARACTER ( KIND = C_CHAR, LEN = strlen( cstr ) ) :: fchar

    INTEGER ( KIND = ip_ ) :: i
    CHARACTER( KIND = C_CHAR ), DIMENSION( : ), POINTER :: temp

    CALL c_f_pointer( cstr, temp, shape = (/ strlen( cstr ) /) )

    DO i = 1, SIZE( temp )
      fchar( i : i ) = temp( i )
    END DO
    RETURN

    END FUNCTION cstr_to_fchar

  END MODULE GALAHAD_common_ciface
