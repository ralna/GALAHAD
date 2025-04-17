! THIS VERSION: GALAHAD 5.2 - 2025-04-17 AT 16:20 GMT.

#ifdef REAL_32
#define GALAHAD_LOAD_ROUTINES_NAME galahad_load_routines_s
#define GALAHAD_UNLOAD_ROUTINES_NAME galahad_unload_routines_s
#define CUTEST_LOAD_ROUTINES_NAME cutest_load_routines_s
#define CUTEST_UNLOAD_ROUTINES_NAME cutest_unload_routines_s
#elif REAL_128
#define GALAHAD_LOAD_ROUTINES_NAME galahad_load_routines_q
#define GALAHAD_UNLOAD_ROUTINES_NAME galahad_unload_routines_q
#define CUTEST_LOAD_ROUTINES_NAME cutest_load_routines_q
#define CUTEST_UNLOAD_ROUTINES_NAME cutest_unload_routines_q
#else
#define GALAHAD_LOAD_ROUTINES_NAME galahad_load_routines
#define GALAHAD_UNLOAD_ROUTINES_NAME galahad_unload_routines
#define CUTEST_LOAD_ROUTINES_NAME cutest_load_routines
#define CUTEST_UNLOAD_ROUTINES_NAME cutest_unload_routines
#endif

SUBROUTINE GALAHAD_LOAD_ROUTINES_NAME(libname)
    USE ISO_C_BINDING, ONLY : C_CHAR
    CHARACTER ( KIND = C_CHAR ), DIMENSION( * ), INTENT( IN ) :: libname
    CALL CUTEST_LOAD_ROUTINES_NAME(libname)
END SUBROUTINE GALAHAD_LOAD_ROUTINES_NAME

SUBROUTINE GALAHAD_UNLOAD_ROUTINES_NAME()
    CALL CUTEST_UNLOAD_ROUTINES_NAME()
END SUBROUTINE GALAHAD_UNLOAD_ROUTINES_NAME
