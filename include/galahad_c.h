#ifdef __cplusplus
extern "C" {
#else
#include <stdbool.h>
#include <stdint.h>
#endif

#ifdef QUAD_REALS_EXIST
#include <quadmath.h>
#endif

// include guard
#ifndef GALAHAD_C_H
#define GALAHAD_C_H

#include "galahad_c_common.h"
#include "galahad_c_single.h"
#include "galahad_c_double.h"
#ifdef QUAD_REALS_EXIST
#include "galahad_c_quadruple.h"
#endif

// end include guard
#endif

#ifdef __cplusplus
} /* extern "C" */
#endif
