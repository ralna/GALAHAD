       char buff[128];
       int nf = quadmath_snprintf(buff, sizeof buff, "%*.2Qf", f);
       if ((size_t) nf < sizeof buff)
          printf("%" i_ipc_ " evaluations. Optimal objective value = %s"
            " status = %1" i_ipc_ "\n", 
          inform.f_eval, buff, inform.status);
//        printf("%" i_ipc_ " evaluations. Optimal objective value = %.2Qf"
//          " status = %1" i_ipc_ "\n", 
//        inform.f_eval, f, x, inform.status);
