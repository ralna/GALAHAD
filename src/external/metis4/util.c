/*
 * Empty dummy version of
 *
 *  util.c
 *
 * in absence of the proper (licence-restricted) MeTiS 4 code
 * 
 * simplified version of util.c by Nick Gould, STFC-RAL, 2024-03 
 * for utilities required for tests
 *
 * If you wish to enable the proper version, and qualify under the
 * licence conditons specified in the LICENSE file, please refer to
 * the AVAILABILITY file.
 *
 * Nick Gould, for GALAHAD productions
 * This version: GALAHAD 5.2 - 2025-03-02
 *
 */

#include "metis4.h"

/* GKlib utilities */

#ifndef DMALLOC

idxtype *idxmalloc(int n, char *msg)
{
  if (n == 0)
    return NULL;

  return (idxtype *)GKmalloc(sizeof(idxtype)*n, msg);
}

void *GKmalloc(int nbytes, char *msg)
{
  void *ptr;

  if (nbytes == 0)
    return NULL;

  ptr = (void *)malloc(nbytes);
  if (ptr == NULL) 
    errexit("***Memory allocation failed for %s. Requested size: %d bytes", msg, nbytes);

  return ptr;
}
#endif

void GK_free(void **ptr1,...)
{
  va_list plist;
  void **ptr;

  if (*ptr1 != NULL)
    free(*ptr1);
  *ptr1 = NULL;

  va_start(plist, ptr1);

  /* while ((int)(ptr = va_arg(plist, void **)) != -1) { */
  while ((ptr = va_arg(plist, void **)) != LTERM) {
    if (*ptr != NULL)
      free(*ptr);
    *ptr = NULL;
  }

  va_end(plist);
}           

void errexit(char *f_str,...)
{
  va_list argp;

  fprintf(stderr, "[METIS Fatal Error] ");

  va_start(argp, f_str);
  vfprintf(stderr, f_str, argp);
  va_end(argp);

  if (strlen(f_str) == 0 || f_str[strlen(f_str)-1] != '\n')
    fprintf(stderr,"\n");
  fflush(stderr);

  abort();
}
