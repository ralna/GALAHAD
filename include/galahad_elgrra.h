/* \file galahad_elgrra.h */

/*
 * assign names for the elfun, group and & routines using the C pre-processor.
 * possibilities are (currently) single (r4 and double (r8, default) reals
 *
 * Nick Gould for GALAHAD
 * initial version, 2023-11-15
 * this version 2024-12-13
 */

#ifdef REAL_32
#define ELFUN_r ELFUN_s
#define ELFUN_flexible_r ELFUN_flexible_s
#define GROUP_r GROUP_s
#define RANGE_r RANGE_s
#elif REAL_128
#define ELFUN_r ELFUN_q
#define ELFUN_flexible_r ELFUN_flexible_q
#define GROUP_r GROUP_q
#define RANGE_r RANGE_q
#else
#define ELFUN_r ELFUN
#define ELFUN_flexible_r ELFUN_flexible
#define GROUP_r GROUP
#define RANGE_r RANGE
#endif
