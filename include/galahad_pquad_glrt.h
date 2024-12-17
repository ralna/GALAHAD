//        char buff[128];
//        int nf = quadmath_snprintf(buff, sizeof buff, 
//                                   "%*.2Qf", inform.obj_regularized);
//        if ((size_t) nf < sizeof buff)
//          printf("MR = %1" i_ipc_ "%1" i_ipc_ 
//                 " glrt_solve_problem exit status = %" i_ipc_ ", f = %s\n", 
//                 unit_m, new_weight, inform.status, buff);
//        printf("MR = %1" i_ipc_ "%1" i_ipc_ 
//               " glrt_solve_problem exit status = %" i_ipc_ ", f = %.2Qf\n", 
//               unit_m, new_weight, inform.status, inform.obj_regularized );
        printf("MR = %1" i_ipc_ "%1" i_ipc_ 
               " glrt_solve_problem exit status = %" i_ipc_ ", f = %.2f\n", 
               unit_m, new_weight, inform.status, 
               (double)inform.obj_regularized );
