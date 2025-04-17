! THIS VERSION: GALAHAD 5.2 - 2025-04-17 AT 16:20 GMT.

#include "galahad_modules.h"

#ifdef REAL_32
#define galahad_load_routines_r galahad_load_routines_s
#define galahad_unload_routines_r galahad_unload_routines_s
#elif REAL_128
#define galahad_load_routines_r galahad_load_routines_q
#define galahad_unload_routines_r galahad_unload_routines_q
#else
#define galahad_load_routines_r galahad_load_routines
#define galahad_unload_routines_r galahad_unload_routines
#endif

SUBROUTINE galahad_load_routines_r(libname)
    USE CUTEST_TRAMPOLINE_precision
    USE ISO_C_BINDING, ONLY : C_CHAR
    CHARACTER ( KIND = C_CHAR ), DIMENSION( * ), INTENT( IN ) :: libname
    CALL CUTEST_LOAD_ROUTINES(libname)
END SUBROUTINE galahad_load_routines_r

SUBROUTINE galahad_unload_routines_r()
    USE CUTEST_TRAMPOLINE_precision
    CALL CUTEST_UNLOAD_ROUTINES()
END SUBROUTINE galahad_unload_routines_r
