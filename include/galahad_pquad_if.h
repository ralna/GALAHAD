//            char buff[128];
//            int nf = quadmath_snprintf(buff, sizeof buff,
//                                       "%*.2Qf", inform.obj);
//            if ((size_t) nf < sizeof buff) 
//              printf("%c:%6" i_ipc_ " cg iterations. Optimal objective " 
//                     "value = %s status = %1" i_ipc_ "\n",
//                     st, inform.cg_iter, buff, inform.status);
//            printf("%c:%6" i_ipc_ " cg iterations. Optimal objective " 
//                   "value = %.2Qf status = %1" i_ipc_ "\n",
//                   st, inform.cg_iter, inform.obj, inform.status);
            printf("%c:%6" i_ipc_ " cg iterations. Optimal objective " 
                   "value = %.2f status = %1" i_ipc_ "\n",
                   st, inform.cg_iter, (double)inform.obj, inform.status);
