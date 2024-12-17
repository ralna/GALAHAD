{
//          char buff[128];
//          int nf = quadmath_snprintf(buff, sizeof buff, "%*.2Qf", inform.obj);
//          if ((size_t) nf < sizeof buff) 
//            printf("format %c: DPS_solve_problem exit status   = %1" i_ipc_ 
//                 ", f = %s\n", st, inform.status, buff );
//        printf("format %c: DPS_solve_problem exit status   = %1" i_ipc_ 
//               ", f = %.2Qf\n", st, inform.status, inform.obj );
        printf("format %c: DPS_solve_problem exit status   = %1" i_ipc_ 
               ", f = %.2f\n", st, inform.status, (double)inform.obj );
}
