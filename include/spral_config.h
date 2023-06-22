/* config.h.  Generated from config.h.in by configure.  */
/* config.h.in.  Generated from configure.ac by autoheader.  */

/* Define to dummy `main' function (if any) required to link to the Fortran
   libraries. */
/* #undef F77_DUMMY_MAIN */

/* Define if F77 and FC dummy `main' functions are identical. */
/* #undef FC_DUMMY_MAIN_EQ_F77 */

/* Define if you have a BLAS library. */
#define HAVE_BLAS 1

/* Define if you have a GTG library. */
/* #undef HAVE_GTG */

/* Define if you have hwloc library */
/* #undef HAVE_HWLOC */
/* #define HAVE_HWLOC 1 */
#ifdef SPRAL_NO_HWLOC
#undef HAVE_HWLOC
#else
#define HAVE_HWLOC 1
#endif

/* Define to 1 if you have the <inttypes.h> header file. */
#define HAVE_INTTYPES_H 1

/* Define if you have LAPACK library. */
#define HAVE_LAPACK 1

/* Define to 1 if you have the <memory.h> header file. */
#define HAVE_MEMORY_H 1

/* Define if you have a MeTiS library. */
#define HAVE_METIS 1

/* Define to 1 if you are compiling against NVCC */
/* #undef HAVE_NVCC */

/* Define to 1 if you have sched_getcpu(). */
/* #define HAVE_SCHED_GETCPU 1 */
/* #undef HAVE_SCHED_GETCPU */
#ifdef SPRAL_NO_SCHED_GETCPU
#undef HAVE_SCHED_GETCPU
#else
#define HAVE_SCHED_GETCPU 1
#endif

/* Define to 1 if you have the <stdint.h> header file. */
#define HAVE_STDINT_H 1

/* Define to 1 if you have the <stdlib.h> header file. */
#define HAVE_STDLIB_H 1

/* Define to 1 if you have std::align(). */
/* #undef HAVE_STD_ALIGN */

/* Define to 1 if you have the <strings.h> header file. */
#define HAVE_STRINGS_H 1

/* Define to 1 if you have the <string.h> header file. */
#define HAVE_STRING_H 1

/* Define to 1 if you have the <sys/stat.h> header file. */
#define HAVE_SYS_STAT_H 1

/* Define to 1 if you have the <sys/types.h> header file. */
#define HAVE_SYS_TYPES_H 1

/* Define to 1 if you have the <unistd.h> header file. */
#define HAVE_UNISTD_H 1

/* Name of package */
#define PACKAGE "spral"

/* Define to the address where bug reports for this package should be sent. */
#define PACKAGE_BUGREPORT "jonathan.hogg@stfc.ac.uk"

/* Define to the full name of this package. */
#define PACKAGE_NAME "spral"

/* Define to the full name and version of this package. */
#define PACKAGE_STRING "spral 2016.06.24"

/* Define to the one symbol short name of this package. */
#define PACKAGE_TARNAME "spral"

/* Define to the home page for this package. */
#define PACKAGE_URL ""

/* Define to the version of this package. */
#define PACKAGE_VERSION "2016.06.24"

/* Define to 1 to enable profiling */
/* #undef PROFILE */

/* Define to 1 if you have the ANSI C header files. */
#define STDC_HEADERS 1

/* Version number of package */
#define VERSION "2016.06.24"
