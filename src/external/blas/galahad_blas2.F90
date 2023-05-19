! THIS VERSION: GALAHAD 5.0 - 2023-05-18 AT 14:15 GMT.

#ifdef GALAHAD_SINGLE
     include "snrm2.f90"
     include "srotg.f90"
#else
     include "dnrm2.f90"
     include "drotg.f90"
#endif
