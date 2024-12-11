#ifdef INTEGER_64
#define hsl_metis galahad_metis_64
#define HSL_METIS galahad_metis_64
#else
#define hsl_metis galahad_metis
#define HSL_METIS galahad_metis
#endif

#ifdef INTEGER_64
#ifdef DUMMY_HSL
#define HSL_KINDS gal_hsl_kinds_64
#define HSL_KINDS_single gal_hsl_kinds_single_64
#define HSL_KINDS_double gal_hsl_kinds_double_64
#define HSL_KINDS_quadruple gal_hsl_kinds_quadruple_64
#define hsl_kinds gal_hsl_kinds_64
#define hsl_kinds_single gal_hsl_kinds_single_64
#define hsl_kinds_double gal_hsl_kinds_double_64
#define hsl_kinds_quadruple gal_hsl_kinds_quadruple_64
#define hsl_kb22_long_integer gal_hsl_kb22_long_integer_64
#define hsl_mc68_integer gal_hsl_mc68_integer_64
#define hsl_mc68_integer_ciface gal_hsl_mc68_integer_64_ciface
#define hsl_mc78_integer gal_hsl_mc78_integer_64
#define hsl_of01_integer gal_hsl_of01_integer_64
#define HSL_OF01_integer gal_hsl_of01_integer_64
#define hsl_zb01_integer gal_hsl_zb01_integer_64
#else
#define HSL_KINDS hsl_kinds_64
#define HSL_KINDS_single hsl_kinds_single_64
#define HSL_KINDS_double hsl_kinds_double_64
#define HSL_KINDS_quadruple hsl_kinds_quadruple_64
#define hsl_kinds hsl_kinds_64
#define hsl_kinds_single hsl_kinds_single_64
#define hsl_kinds_double hsl_kinds_double_64
#define hsl_kinds_quadruple hsl_kinds_quadruple_64
#define hsl_kb22_long_integer hsl_kb22_long_integer_64
#define hsl_mc68_integer hsl_mc68_integer_64
#define hsl_mc68_integer_ciface hsl_mc68_integer_64_ciface
#define hsl_mc78_integer hsl_mc78_integer_64
#define hsl_of01_integer hsl_of01_integer_64
#define HSL_OF01_integer hsl_of01_integer_64
#define hsl_zb01_integer hsl_zb01_integer_64
#endif
#else
#ifdef DUMMY_HSL
#define HSL_KINDS gal_hsl_kinds
#define HSL_KINDS_single gal_hsl_kinds_single
#define HSL_KINDS_double gal_hsl_kinds_double
#define HSL_KINDS_quadruple gal_hsl_kinds_quadruple
#define hsl_kinds gal_hsl_kinds
#define hsl_kinds_single gal_hsl_kinds_single
#define hsl_kinds_double gal_hsl_kinds_double
#define hsl_kinds_quadruple gal_hsl_kinds_quadruple
#define hsl_kb22_long_integer gal_hsl_kb22_long_integer
#define hsl_mc68_integer gal_hsl_mc68_integer
#define hsl_mc68_integer_ciface gal_hsl_mc68_integer_ciface
#define hsl_mc78_integer gal_hsl_mc78_integer
#define hsl_of01_integer gal_hsl_of01_integer
#define HSL_OF01_integer gal_hsl_of01_integer
#define hsl_zb01_integer gal_hsl_zb01_integer
#endif
#endif

#ifdef INTEGER_64
#ifdef DUMMY_HSL
#define KB07AI GAL_KB07AI_64
#define kb07ai gal_kb07ai_64
#define KB21AI GAL_KB21AI_64
#define KB21BI GAL_KB21BI_64
#define KB21CI GAL_KB21CI_64
#define KB21DI GAL_KB21DI_64
#define KB21EI GAL_KB21EI_64
#define KB21FI GAL_KB21FI_64
#define KB21GI GAL_KB21GI_64
#define KB21HI GAL_KB21HI_64
#else
#define KB07AI KB07AI_64
#define kb07ai kb07ai_64
#define KB21AI KB21AI_64
#define KB21BI KB21BI_64
#define KB21CI KB21CI_64
#define KB21DI KB21DI_64
#define KB21EI KB21EI_64
#define KB21FI KB21FI_64
#define KB21GI KB21GI_64
#define KB21HI KB21HI_64
#endif
#else
#ifdef DUMMY_HSL
#define KB07AI GAL_KB07AI
#define kb07ai gal_kb07ai
#define KB21AI GAL_KB21AI
#define KB21BI GAL_KB21BI
#define KB21CI GAL_KB21CI
#define KB21DI GAL_KB21DI
#define KB21EI GAL_KB21EI
#define KB21FI GAL_KB21FI
#define KB21GI GAL_KB21GI
#define KB21HI GAL_KB21HI
#endif
#endif

#ifdef INTEGER_64
#ifdef NO_UNDERSCORE_INTEGER_64
#define idamax idamax64
#elif DOUBLE_UNDERSCORE_INTEGER_64
#define idamax idamax__64
#elif NO_SYMBOL_INTEGER_64
#else
#define idamax idamax_64
#endif
#endif

#ifdef REAL_32
#include "hsl_subset_single.h"
#else
#ifdef REAL_128
#include "hsl_subset_quadruple.h"
#else
#include "hsl_subset_double.h"
#endif
#endif

