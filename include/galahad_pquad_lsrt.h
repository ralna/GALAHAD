//    char buff[128];
//    int nf = quadmath_snprintf(buff, sizeof buff, "%*.2Qf", inform.obj);
//    if ((size_t) nf < sizeof buff) 
//      printf("lsrt_solve_problem exit status = %" i_ipc_
//             ", f = %s\n", inform.status, buff );
//    printf("lsrt_solve_problem exit status = %" i_ipc_
//           ", f = %.2Qf\n", inform.status, inform.obj );
    printf("lsrt_solve_problem exit status = %" i_ipc_
           ", f = %.2f\n", inform.status, (double)inform.obj );
