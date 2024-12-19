//        char buff[128];
//        int nf = quadmath_snprintf(buff, sizeof buff, "%*.2Qf", inform.obj);
//        if ((size_t) nf < sizeof buff)
//          printf("MR = %1" i_ipc_ "%1" i_ipc_ 
//                 " gltr_solve_problem exit status = %" i_ipc_ ", f = %s\n", 
//                 unit_m, new_radius, inform.status, buff );
//        printf("MR = %1" i_ipc_ "%1" i_ipc_ 
//               " gltr_solve_problem exit status = %" i_ipc_ ", f = %.2Qf\n", 
//               unit_m, new_radius, inform.status, inform.obj );
        printf("MR = %1" i_ipc_ "%1" i_ipc_ 
               " gltr_solve_problem exit status = %" i_ipc_ ", f = %.2f\n", 
               unit_m, new_radius, inform.status, (double)inform.obj );
