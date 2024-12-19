//            char buff[128];
//            int nf = quadmath_snprintf(buff, sizeof buff,
//                                       "%*.2Qf", inform.r_norm);
//            if ((size_t) nf < sizeof buff) 
//              printf("storage type %c%1" i_ipc_ ":  status = %1" i_ipc_ 
//                    ", ||r|| = %s\n", st, use_s, inform.status, buff );
//             printf("storage type %c%1" i_ipc_ ":  status = %1" i_ipc_ 
//                    ", ||r|| = %5.2Qf\n",
//                    st, use_s, inform.status, inform.r_norm );
             printf("storage type %c%1" i_ipc_ ":  status = %1" i_ipc_ 
                    ", ||r|| = %5.2f\n",
                    st, use_s, inform.status, (double)inform.r_norm );
