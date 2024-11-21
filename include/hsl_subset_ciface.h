#ifdef REAL_32
#include "hsl_subset_ciface_single.h"
#else
#ifdef REAL_128
#include "hsl_subset_ciface_quadruple.h"
#else
#include "hsl_subset_ciface_double.h"
#endif
#endif

#ifdef INTEGER_64
#define mc68_control_i mc68_control_i_64
#define mc68_info_i mc68_info_i_64
#define mc68_default_control_i mc68_default_control_i_64
#define mc68_order_i mc68_order_i_64
#endif
