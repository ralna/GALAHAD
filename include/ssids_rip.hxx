/** \file
 *  \copyright 2024 GALAHAD productions
 *  \licence   BSD licence, see LICENCE file for details
 *  \author    Nick Gould
 *  \version   GALAHAD 4.3 - 2024-02-18 AT 08:30 GMT
 */

#include <stdint.h>

/* real precision employed */

#ifdef SINGLE
typedef float rpc_;
#else
typedef double rpc_;
#endif

/* integer storage employed */

#ifdef INTEGER_64
typedef int64_t ipc_;
typedef uint64_t uipc_;
#else
typedef int ipc_;
typedef unsigned int uipc_;
#endif

/* generic storage */

typedef int64_t longc_;
