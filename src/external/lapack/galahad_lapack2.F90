! THIS VERSION: GALAHAD 5.0 - 2023-05-18 AT 14:15 GMT.

#ifdef GALAHAD_SINGLE
     include "slassq.f90"
     include "slartg.f90"
#else
     include "dlassq.f90"
     include "dlartg.f90"
#endif
