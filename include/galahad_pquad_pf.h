//            char buff[128];
//            int nf = quadmath_snprintf(buff, sizeof buff,
//                                       "%*.2Qf", inform.obj);
//            if ((size_t) nf < sizeof buff) 
//              printf("P%1" d_ipc_ ":%6" d_ipc_ " iterations. Optimal objective "
//                     "value = %s status = %1" d_ipc_ "\n",
//                    model, inform.iter, buff, inform.status);
//            printf("P%1" d_ipc_ ":%6" d_ipc_ " iterations. Optimal objective "
//                   "value = %.2Qf status = %1" d_ipc_ "\n",
//                   model, inform.iter, inform.obj, inform.status);
            printf("P%1" d_ipc_ ":%6" d_ipc_ " iterations. Optimal objective "
                   "value = %.2f status = %1" d_ipc_ "\n",
                   model, inform.iter, (double)inform.obj, inform.status);
