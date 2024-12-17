//       char buff[128];
//       int nf = quadmath_snprintf(buff, sizeof buff, "%*.2Qf", inform.obj);
//       if ((size_t) nf < sizeof buff)
//          printf("%c:%6" i_ipc_ " evaluations. Optimal objective "
//                 "value = %s status = %1" i_ipc_ "\n", 
//                 st, inform.f_eval, buff, inform.status);
//            printf("%c:%6" i_ipc_ " evaluations. Optimal objective "
//                   "value = %.2Qf status = %1" i_ipc_ "\n", 
//                   st, inform.f_eval, inform.obj, inform.status);
            printf("%c:%6" i_ipc_ " evaluations. Optimal objective "
                   "value = %.2f status = %1" i_ipc_ "\n", 
                   st, inform.f_eval, (double)inform.obj, inform.status);
