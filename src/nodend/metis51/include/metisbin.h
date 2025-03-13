/*
 * metisbin.h
 *
 * This file contains the various header inclusions
 *
 * Started 8/9/02
 * George
 */

#include <GKlib.h>
#include <stddef.h>
#include <stdlib.h>
#include <stdarg.h>
#include <stdio.h>
#include <ctype.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <limits.h>
#include <signal.h>
#include <setjmp.h>
#include <assert.h>


#if defined(ENABLE_OPENMP)
  #include <omp.h>
#endif


#include <metis.h>
#include "rename.h"
#include "gklib_defs.h"
#include "defs.h"
#include "macros.h"
#include "defs.h"
#include "struct.h"
#include "proto.h"


#if defined(COMPILER_GCC)
extern char* strdup (const char *);
#endif

#if defined(COMPILER_MSC)
#if defined(rint)
  #undef rint
#endif
#define rint(x) ((idx_t)((x)+0.5))  /* MSC does not have rint() function */
#define __func__ "dummy-function"
#endif

typedef struct {
  idx_t ptype;
  idx_t objtype;
  idx_t ctype;
  idx_t iptype;
  idx_t rtype;

  idx_t no2hop;
  idx_t minconn;
  idx_t contig;

  idx_t nooutput;

  idx_t balance;
  idx_t ncuts;
  idx_t niter;

  idx_t gtype;
  idx_t ncommon;

  idx_t seed;
  idx_t dbglvl;

  idx_t nparts;

  idx_t nseps;
  idx_t ufactor;
  idx_t pfactor;
  idx_t compress;
  idx_t ccorder;

  char *filename;
  char *outfile;
  char *xyzfile;
  char *tpwgtsfile;
  char *ubvecstr;

  idx_t wgtflag;
  idx_t numflag;
  real_t *tpwgts;
  real_t *ubvec;

  real_t iotimer;
  real_t parttimer;
  real_t reporttimer;

  size_t maxmemory;
} params_t;

#include "protobin.h"
#include "defsbin.h"
