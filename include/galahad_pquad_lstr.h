//        char buff[128];
//        int nf = quadmath_snprintf(buff, sizeof buff, "%*.2Qf", inform.r_norm);
//        if ((size_t) nf < sizeof buff) 
//          printf("%1" i_ipc_ " lstr_solve_problem exit status = %" i_ipc_
//                 ", f = %s\n", new_radius, inform.status, buff );
//      printf("%1" i_ipc_ " lstr_solve_problem exit status = %" i_ipc_
//             ", f = %.2Qf\n", new_radius, inform.status, inform.r_norm );
      printf("%1" i_ipc_ " lstr_solve_problem exit status = %" i_ipc_
             ", f = %.2f\n", new_radius, inform.status, (double)inform.r_norm );
