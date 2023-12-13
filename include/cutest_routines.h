/* \file cutest_routines.h */

/*
 * assign names for each CUTEst routine using the C pre-processor.
 * possibilities are (currently) single (r4 and double (r8, default) reals
 *
 * Nick Gould for CUTEst
 * initial version, 2023-11-11
 * this version 2023-11-11
 */

#ifdef CUTEST_SINGLE
#include "cutest_routines_single.h"
#else
#include "cutest_routines_double.h"
#endif
