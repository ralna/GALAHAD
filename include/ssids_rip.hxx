/** \file
 *  \copyright 2024 GALAHAD productions
 *  \licence   BSD licence, see LICENCE file for details
 *  \author    Nick Gould
 *  \version   GALAHAD 4.3 - 2024-02-03 AT 10:30 GMT
 */

#include <stdint.h>

/* real precision employed */

#ifdef SPRAL_SINGLE
#define rpc_ float
#else
#define rpc_ double
#endif

/* integer storage employed */

#ifdef INTEGER_64
#define ipc_ int64_t
#define uipc_ uint64_t
#else
#define ipc_ int
#define uipc_ unsigned int
#endif

/* generic storage */

#define longc_ int64_t
