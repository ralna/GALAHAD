//* \file dqp_pyiface.c */

/*
 * THIS VERSION: GALAHAD 5.0 - 2024-06-15 AT 11:50 GMT.
 *
 *-*-*-*-*-*-*-*-*-  GALAHAD_DQP PYTHON INTERFACE  *-*-*-*-*-*-*-*-*-*-
 *
 *  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
 *  Principal author: Jaroslav Fowkes & Nick Gould
 *
 *  History -
 *   originally released GALAHAD Version 4.1. March 23rd 2023
 *
 *  For full documentation, see
 *   http://galahad.rl.ac.uk/galahad-www/specs.html
 */

#include "galahad_python.h"
#include "galahad_dqp.h"

/* Nested FDC, SLS, SBLS, GLTR, SCU and RPD control and inform prototypes */
bool fdc_update_control(struct fdc_control_type *control,
                        PyObject *py_options);
PyObject* fdc_make_options_dict(const struct fdc_control_type *control);
PyObject* fdc_make_inform_dict(const struct fdc_inform_type *inform);
bool sls_update_control(struct sls_control_type *control,
                         PyObject *py_options);
PyObject* sls_make_options_dict(const struct sls_control_type *control);
PyObject* sls_make_inform_dict(const struct sls_inform_type *inform);
bool sbls_update_control(struct sbls_control_type *control,
                         PyObject *py_options);
PyObject* sbls_make_options_dict(const struct sbls_control_type *control);
PyObject* sbls_make_inform_dict(const struct sbls_inform_type *inform);
bool gltr_update_control(struct gltr_control_type *control,
                         PyObject *py_options);
PyObject* gltr_make_options_dict(const struct gltr_control_type *control);
PyObject* gltr_make_inform_dict(const struct gltr_inform_type *inform);
PyObject* scu_make_inform_dict(const struct scu_inform_type *inform);
PyObject* rpd_make_inform_dict(const struct rpd_inform_type *inform);

/* Module global variables */
static void *data;                       // private internal data
static struct dqp_control_type control;  // control struct
static struct dqp_inform_type inform;    // inform struct
static bool init_called = false;         // record if initialise was called
static int status = 0;                   // exit status

//  *-*-*-*-*-*-*-*-*-*-   UPDATE CONTROL    -*-*-*-*-*-*-*-*-*-*

/* Update the control options: use C defaults but update any passed via Python*/
static bool dqp_update_control(struct dqp_control_type *control,
                               PyObject *py_options){

    // Use C defaults if Python options not passed
    if(!py_options) return true;

    PyObject *key, *value;
    Py_ssize_t pos = 0;
    const char* key_name;

    // Iterate over Python options dictionary
    while(PyDict_Next(py_options, &pos, &key, &value)) {

        // Parse options key
        if(!parse_options_key(key, &key_name))
            return false;

        // Parse each option
        if(strcmp(key_name, "error") == 0){
            if(!parse_int_option(value, "error",
                                  &control->error))
                return false;
            continue;
        }
        if(strcmp(key_name, "out") == 0){
            if(!parse_int_option(value, "out",
                                  &control->out))
                return false;
            continue;
        }
        if(strcmp(key_name, "print_level") == 0){
            if(!parse_int_option(value, "print_level",
                                  &control->print_level))
                return false;
            continue;
        }
        if(strcmp(key_name, "start_print") == 0){
            if(!parse_int_option(value, "start_print",
                                  &control->start_print))
                return false;
            continue;
        }
        if(strcmp(key_name, "stop_print") == 0){
            if(!parse_int_option(value, "stop_print",
                                  &control->stop_print))
                return false;
            continue;
        }
        if(strcmp(key_name, "print_gap") == 0){
            if(!parse_int_option(value, "print_gap",
                                  &control->print_gap))
                return false;
            continue;
        }
        if(strcmp(key_name, "dual_starting_point") == 0){
            if(!parse_int_option(value, "dual_starting_point",
                                  &control->dual_starting_point))
                return false;
            continue;
        }
        if(strcmp(key_name, "maxit") == 0){
            if(!parse_int_option(value, "maxit",
                                  &control->maxit))
                return false;
            continue;
        }
        if(strcmp(key_name, "max_sc") == 0){
            if(!parse_int_option(value, "max_sc",
                                  &control->max_sc))
                return false;
            continue;
        }
        if(strcmp(key_name, "cauchy_only") == 0){
            if(!parse_int_option(value, "cauchy_only",
                                  &control->cauchy_only))
                return false;
            continue;
        }
        if(strcmp(key_name, "arc_search_maxit") == 0){
            if(!parse_int_option(value, "arc_search_maxit",
                                  &control->arc_search_maxit))
                return false;
            continue;
        }
        if(strcmp(key_name, "cg_maxit") == 0){
            if(!parse_int_option(value, "cg_maxit",
                                  &control->cg_maxit))
                return false;
            continue;
        }
        if(strcmp(key_name, "explore_optimal_subspace") == 0){
            if(!parse_int_option(value, "explore_optimal_subspace",
                                  &control->explore_optimal_subspace))
                return false;
            continue;
        }
        if(strcmp(key_name, "restore_problem") == 0){
            if(!parse_int_option(value, "restore_problem",
                                  &control->restore_problem))
                return false;
            continue;
        }
        if(strcmp(key_name, "sif_file_device") == 0){
            if(!parse_int_option(value, "sif_file_device",
                                  &control->sif_file_device))
                return false;
            continue;
        }
        if(strcmp(key_name, "qplib_file_device") == 0){
            if(!parse_int_option(value, "qplib_file_device",
                                  &control->qplib_file_device))
                return false;
            continue;
        }
        if(strcmp(key_name, "rho") == 0){
            if(!parse_double_option(value, "rho",
                                  &control->rho))
                return false;
            continue;
        }
        if(strcmp(key_name, "infinity") == 0){
            if(!parse_double_option(value, "infinity",
                                  &control->infinity))
                return false;
            continue;
        }
        if(strcmp(key_name, "stop_abs_p") == 0){
            if(!parse_double_option(value, "stop_abs_p",
                                  &control->stop_abs_p))
                return false;
            continue;
        }
        if(strcmp(key_name, "stop_rel_p") == 0){
            if(!parse_double_option(value, "stop_rel_p",
                                  &control->stop_rel_p))
                return false;
            continue;
        }
        if(strcmp(key_name, "stop_abs_d") == 0){
            if(!parse_double_option(value, "stop_abs_d",
                                  &control->stop_abs_d))
                return false;
            continue;
        }
        if(strcmp(key_name, "stop_rel_d") == 0){
            if(!parse_double_option(value, "stop_rel_d",
                                  &control->stop_rel_d))
                return false;
            continue;
        }
        if(strcmp(key_name, "stop_abs_c") == 0){
            if(!parse_double_option(value, "stop_abs_c",
                                  &control->stop_abs_c))
                return false;
            continue;
        }
        if(strcmp(key_name, "stop_rel_c") == 0){
            if(!parse_double_option(value, "stop_rel_c",
                                  &control->stop_rel_c))
                return false;
            continue;
        }
        if(strcmp(key_name, "stop_cg_relative") == 0){
            if(!parse_double_option(value, "stop_cg_relative",
                                  &control->stop_cg_relative))
                return false;
            continue;
        }
        if(strcmp(key_name, "stop_cg_absolute") == 0){
            if(!parse_double_option(value, "stop_cg_absolute",
                                  &control->stop_cg_absolute))
                return false;
            continue;
        }
        if(strcmp(key_name, "cg_zero_curvature") == 0){
            if(!parse_double_option(value, "cg_zero_curvature",
                                  &control->cg_zero_curvature))
                return false;
            continue;
        }
        if(strcmp(key_name, "max_growth") == 0){
            if(!parse_double_option(value, "max_growth",
                                  &control->max_growth))
                return false;
            continue;
        }
        if(strcmp(key_name, "identical_bounds_tol") == 0){
            if(!parse_double_option(value, "identical_bounds_tol",
                                  &control->identical_bounds_tol))
                return false;
            continue;
        }
        if(strcmp(key_name, "cpu_time_limit") == 0){
            if(!parse_double_option(value, "cpu_time_limit",
                                  &control->cpu_time_limit))
                return false;
            continue;
        }
        if(strcmp(key_name, "clock_time_limit") == 0){
            if(!parse_double_option(value, "clock_time_limit",
                                  &control->clock_time_limit))
                return false;
            continue;
        }
        if(strcmp(key_name, "initial_perturbation") == 0){
            if(!parse_double_option(value, "initial_perturbation",
                                  &control->initial_perturbation))
                return false;
            continue;
        }
        if(strcmp(key_name, "perturbation_reduction") == 0){
            if(!parse_double_option(value, "perturbation_reduction",
                                  &control->perturbation_reduction))
                return false;
            continue;
        }
        if(strcmp(key_name, "final_perturbation") == 0){
            if(!parse_double_option(value, "final_perturbation",
                                  &control->final_perturbation))
                return false;
            continue;
        }
        if(strcmp(key_name, "factor_optimal_matrix") == 0){
            if(!parse_bool_option(value, "factor_optimal_matrix",
                                  &control->factor_optimal_matrix))
                return false;
            continue;
        }
        if(strcmp(key_name, "remove_dependencies") == 0){
            if(!parse_bool_option(value, "remove_dependencies",
                                  &control->remove_dependencies))
                return false;
            continue;
        }
        if(strcmp(key_name, "treat_zero_bounds_as_general") == 0){
            if(!parse_bool_option(value, "treat_zero_bounds_as_general",
                                  &control->treat_zero_bounds_as_general))
                return false;
            continue;
        }
        if(strcmp(key_name, "exact_arc_search") == 0){
            if(!parse_bool_option(value, "exact_arc_search",
                                  &control->exact_arc_search))
                return false;
            continue;
        }
        if(strcmp(key_name, "subspace_direct") == 0){
            if(!parse_bool_option(value, "subspace_direct",
                                  &control->subspace_direct))
                return false;
            continue;
        }
        if(strcmp(key_name, "subspace_alternate") == 0){
            if(!parse_bool_option(value, "subspace_alternate",
                                  &control->subspace_alternate))
                return false;
            continue;
        }
        if(strcmp(key_name, "subspace_arc_search") == 0){
            if(!parse_bool_option(value, "subspace_arc_search",
                                  &control->subspace_arc_search))
                return false;
            continue;
        }
        if(strcmp(key_name, "space_critical") == 0){
            if(!parse_bool_option(value, "space_critical",
                                  &control->space_critical))
                return false;
            continue;
        }
        if(strcmp(key_name, "deallocate_error_fatal") == 0){
            if(!parse_bool_option(value, "deallocate_error_fatal",
                                  &control->deallocate_error_fatal))
                return false;
            continue;
        }
        if(strcmp(key_name, "generate_sif_file") == 0){
            if(!parse_bool_option(value, "generate_sif_file",
                                  &control->generate_sif_file))
                return false;
            continue;
        }
        if(strcmp(key_name, "generate_qplib_file") == 0){
            if(!parse_bool_option(value, "generate_qplib_file",
                                  &control->generate_qplib_file))
                return false;
            continue;
        }
        if(strcmp(key_name, "symmetric_linear_solver") == 0){
            if(!parse_char_option(value, "symmetric_linear_solver",
                                  control->symmetric_linear_solver,
                                  sizeof(control->symmetric_linear_solver)))
                return false;
            continue;
        }
        if(strcmp(key_name, "definite_linear_solver") == 0){
            if(!parse_char_option(value, "definite_linear_solver",
                                  control->definite_linear_solver,
                                  sizeof(control->definite_linear_solver)))
                return false;
            continue;
        }
        if(strcmp(key_name, "unsymmetric_linear_solver") == 0){
            if(!parse_char_option(value, "unsymmetric_linear_solver",
                                  control->unsymmetric_linear_solver,
                                  sizeof(control->unsymmetric_linear_solver)))
                return false;
            continue;
        }
        if(strcmp(key_name, "sif_file_name") == 0){
            if(!parse_char_option(value, "sif_file_name",
                                  control->sif_file_name,
                                  sizeof(control->sif_file_name)))
                return false;
            continue;
        }
        if(strcmp(key_name, "qplib_file_name") == 0){
            if(!parse_char_option(value, "qplib_file_name",
                                  control->qplib_file_name,
                                  sizeof(control->qplib_file_name)))
                return false;
            continue;
        }
        if(strcmp(key_name, "prefix") == 0){
            if(!parse_char_option(value, "prefix",
                                  control->prefix,
                                  sizeof(control->prefix)))
                return false;
            continue;
        }
        if(strcmp(key_name, "fdc_options") == 0){
            if(!fdc_update_control(&control->fdc_control, value))
                return false;
            continue;
        }
        if(strcmp(key_name, "sls_options") == 0){
            if(!sls_update_control(&control->sls_control, value))
                return false;
            continue;
        }
        if(strcmp(key_name, "sbls_options") == 0){
            if(!sbls_update_control(&control->sbls_control, value))
                return false;
            continue;
        }
        //if(strcmp(key_name, "gltr_options") == 0){
        //    if(!gltr_update_control(&control->gltr_control, value))
        //        return false;
        //    continue;
        //}
        // Otherwise unrecognised option
        PyErr_Format(PyExc_ValueError,
          "unrecognised option options['%s']\n", key_name);
        return false;
    }

    return true; // success
}

//  *-*-*-*-*-*-*-*-*-*-   MAKE OPTIONS    -*-*-*-*-*-*-*-*-*-*

/* Take the control struct from C and turn it into a python options dict */
// NB not static as it is used for nested inform within QP Python interface
PyObject* dqp_make_options_dict(const struct dqp_control_type *control){
    PyObject *py_options = PyDict_New();

    PyDict_SetItemString(py_options, "error",
                         PyLong_FromLong(control->error));
    PyDict_SetItemString(py_options, "out",
                         PyLong_FromLong(control->out));
    PyDict_SetItemString(py_options, "print_level",
                         PyLong_FromLong(control->print_level));
    PyDict_SetItemString(py_options, "start_print",
                         PyLong_FromLong(control->start_print));
    PyDict_SetItemString(py_options, "stop_print",
                         PyLong_FromLong(control->stop_print));
    PyDict_SetItemString(py_options, "print_gap",
                         PyLong_FromLong(control->print_gap));
    PyDict_SetItemString(py_options, "dual_starting_point",
                         PyLong_FromLong(control->dual_starting_point));
    PyDict_SetItemString(py_options, "maxit",
                         PyLong_FromLong(control->maxit));
    PyDict_SetItemString(py_options, "max_sc",
                         PyLong_FromLong(control->max_sc));
    PyDict_SetItemString(py_options, "cauchy_only",
                         PyLong_FromLong(control->cauchy_only));
    PyDict_SetItemString(py_options, "arc_search_maxit",
                         PyLong_FromLong(control->arc_search_maxit));
    PyDict_SetItemString(py_options, "cg_maxit",
                         PyLong_FromLong(control->cg_maxit));
    PyDict_SetItemString(py_options, "explore_optimal_subspace",
                         PyLong_FromLong(control->explore_optimal_subspace));
    PyDict_SetItemString(py_options, "restore_problem",
                         PyLong_FromLong(control->restore_problem));
    PyDict_SetItemString(py_options, "sif_file_device",
                         PyLong_FromLong(control->sif_file_device));
    PyDict_SetItemString(py_options, "qplib_file_device",
                         PyLong_FromLong(control->qplib_file_device));
    PyDict_SetItemString(py_options, "rho",
                         PyFloat_FromDouble(control->rho));
    PyDict_SetItemString(py_options, "infinity",
                         PyFloat_FromDouble(control->infinity));
    PyDict_SetItemString(py_options, "stop_abs_p",
                         PyFloat_FromDouble(control->stop_abs_p));
    PyDict_SetItemString(py_options, "stop_rel_p",
                         PyFloat_FromDouble(control->stop_rel_p));
    PyDict_SetItemString(py_options, "stop_abs_d",
                         PyFloat_FromDouble(control->stop_abs_d));
    PyDict_SetItemString(py_options, "stop_rel_d",
                         PyFloat_FromDouble(control->stop_rel_d));
    PyDict_SetItemString(py_options, "stop_abs_c",
                         PyFloat_FromDouble(control->stop_abs_c));
    PyDict_SetItemString(py_options, "stop_rel_c",
                         PyFloat_FromDouble(control->stop_rel_c));
    PyDict_SetItemString(py_options, "stop_cg_relative",
                         PyFloat_FromDouble(control->stop_cg_relative));
    PyDict_SetItemString(py_options, "stop_cg_absolute",
                         PyFloat_FromDouble(control->stop_cg_absolute));
    PyDict_SetItemString(py_options, "cg_zero_curvature",
                         PyFloat_FromDouble(control->cg_zero_curvature));
    PyDict_SetItemString(py_options, "max_growth",
                         PyFloat_FromDouble(control->max_growth));
    PyDict_SetItemString(py_options, "identical_bounds_tol",
                         PyFloat_FromDouble(control->identical_bounds_tol));
    PyDict_SetItemString(py_options, "cpu_time_limit",
                         PyFloat_FromDouble(control->cpu_time_limit));
    PyDict_SetItemString(py_options, "clock_time_limit",
                         PyFloat_FromDouble(control->clock_time_limit));
    PyDict_SetItemString(py_options, "initial_perturbation",
                         PyFloat_FromDouble(control->initial_perturbation));
    PyDict_SetItemString(py_options, "perturbation_reduction",
                         PyFloat_FromDouble(control->perturbation_reduction));
    PyDict_SetItemString(py_options, "final_perturbation",
                         PyFloat_FromDouble(control->final_perturbation));
    PyDict_SetItemString(py_options, "factor_optimal_matrix",
                         PyBool_FromLong(control->factor_optimal_matrix));
    PyDict_SetItemString(py_options, "remove_dependencies",
                         PyBool_FromLong(control->remove_dependencies));
    PyDict_SetItemString(py_options, "treat_zero_bounds_as_general",
                         PyBool_FromLong(control->treat_zero_bounds_as_general));
    PyDict_SetItemString(py_options, "exact_arc_search",
                         PyBool_FromLong(control->exact_arc_search));
    PyDict_SetItemString(py_options, "subspace_direct",
                         PyBool_FromLong(control->subspace_direct));
    PyDict_SetItemString(py_options, "subspace_alternate",
                         PyBool_FromLong(control->subspace_alternate));
    PyDict_SetItemString(py_options, "subspace_arc_search",
                         PyBool_FromLong(control->subspace_arc_search));
    PyDict_SetItemString(py_options, "space_critical",
                         PyBool_FromLong(control->space_critical));
    PyDict_SetItemString(py_options, "deallocate_error_fatal",
                         PyBool_FromLong(control->deallocate_error_fatal));
    PyDict_SetItemString(py_options, "generate_sif_file",
                         PyBool_FromLong(control->generate_sif_file));
    PyDict_SetItemString(py_options, "generate_qplib_file",
                         PyBool_FromLong(control->generate_qplib_file));
    PyDict_SetItemString(py_options, "symmetric_linear_solver",
                         PyUnicode_FromString(control->symmetric_linear_solver));
    PyDict_SetItemString(py_options, "definite_linear_solver",
                         PyUnicode_FromString(control->definite_linear_solver));
    PyDict_SetItemString(py_options, "unsymmetric_linear_solver",
                         PyUnicode_FromString(control->unsymmetric_linear_solver));
    PyDict_SetItemString(py_options, "sif_file_name",
                         PyUnicode_FromString(control->sif_file_name));
    PyDict_SetItemString(py_options, "qplib_file_name",
                         PyUnicode_FromString(control->qplib_file_name));
    PyDict_SetItemString(py_options, "prefix",
                         PyUnicode_FromString(control->prefix));
    PyDict_SetItemString(py_options, "fdc_options",
                         fdc_make_options_dict(&control->fdc_control));
    PyDict_SetItemString(py_options, "sls_options",
                         sls_make_options_dict(&control->sls_control));
    PyDict_SetItemString(py_options, "sbls_options",
                         sbls_make_options_dict(&control->sbls_control));
    //PyDict_SetItemString(py_options, "gltr_options",
    //                     gltr_make_options_dict(&control->gltr_control));

    return py_options;
}

//  *-*-*-*-*-*-*-*-*-*-   MAKE TIME    -*-*-*-*-*-*-*-*-*-*

/* Take the time struct from C and turn it into a python dictionary */
static PyObject* dqp_make_time_dict(const struct dqp_time_type *time){
    PyObject *py_time = PyDict_New();

    // Set float/double time entries

    PyDict_SetItemString(py_time, "total",
                         PyFloat_FromDouble(time->total));
    PyDict_SetItemString(py_time, "preprocess",
                         PyFloat_FromDouble(time->preprocess));
    PyDict_SetItemString(py_time, "find_dependent",
                         PyFloat_FromDouble(time->find_dependent));
    PyDict_SetItemString(py_time, "analyse",
                         PyFloat_FromDouble(time->analyse));
    PyDict_SetItemString(py_time, "factorize",
                         PyFloat_FromDouble(time->factorize));
    PyDict_SetItemString(py_time, "solve",
                         PyFloat_FromDouble(time->solve));
    PyDict_SetItemString(py_time, "search",
                         PyFloat_FromDouble(time->search));
    PyDict_SetItemString(py_time, "clock_total",
                         PyFloat_FromDouble(time->clock_total));
    PyDict_SetItemString(py_time, "clock_preprocess",
                         PyFloat_FromDouble(time->clock_preprocess));
    PyDict_SetItemString(py_time, "clock_find_dependent",
                         PyFloat_FromDouble(time->clock_find_dependent));
    PyDict_SetItemString(py_time, "clock_analyse",
                         PyFloat_FromDouble(time->clock_analyse));
    PyDict_SetItemString(py_time, "clock_factorize",
                         PyFloat_FromDouble(time->clock_factorize));
    PyDict_SetItemString(py_time, "clock_solve",
                         PyFloat_FromDouble(time->clock_solve));
    PyDict_SetItemString(py_time, "clock_search",
                         PyFloat_FromDouble(time->clock_search));

    return py_time;
}

//  *-*-*-*-*-*-*-*-*-*-   MAKE INFORM    -*-*-*-*-*-*-*-*-*-*

/* Take the inform struct from C and turn it into a python dictionary */
static PyObject* dqp_make_inform_dict(const struct dqp_inform_type *inform){
    PyObject *py_inform = PyDict_New();

    PyDict_SetItemString(py_inform, "status",
                         PyLong_FromLong(inform->status));
    PyDict_SetItemString(py_inform, "alloc_status",
                         PyLong_FromLong(inform->alloc_status));
    PyDict_SetItemString(py_inform, "bad_alloc",
                         PyUnicode_FromString(inform->bad_alloc));
    PyDict_SetItemString(py_inform, "iter",
                         PyLong_FromLong(inform->iter));
    PyDict_SetItemString(py_inform, "cg_iter",
                         PyLong_FromLong(inform->cg_iter));
    PyDict_SetItemString(py_inform, "factorization_status",
                         PyLong_FromLong(inform->factorization_status));
    PyDict_SetItemString(py_inform, "factorization_integer",
                         PyLong_FromLong(inform->factorization_integer));
    PyDict_SetItemString(py_inform, "factorization_real",
                         PyLong_FromLong(inform->factorization_real));
    PyDict_SetItemString(py_inform, "nfacts",
                         PyLong_FromLong(inform->nfacts));
    PyDict_SetItemString(py_inform, "threads",
                         PyLong_FromLong(inform->threads));
    PyDict_SetItemString(py_inform, "obj",
                         PyFloat_FromDouble(inform->obj));
    PyDict_SetItemString(py_inform, "primal_infeasibility",
                         PyFloat_FromDouble(inform->primal_infeasibility));
    PyDict_SetItemString(py_inform, "dual_infeasibility",
                         PyFloat_FromDouble(inform->dual_infeasibility));
    PyDict_SetItemString(py_inform, "complementary_slackness",
                         PyFloat_FromDouble(inform->complementary_slackness));
    PyDict_SetItemString(py_inform, "non_negligible_pivot",
                         PyFloat_FromDouble(inform->non_negligible_pivot));
    PyDict_SetItemString(py_inform, "feasible",
                         PyBool_FromLong(inform->feasible));

    // include checkpoint arrays
    npy_intp cdim[] = {16};
    PyArrayObject *py_iter =
      (PyArrayObject*) PyArray_SimpleNew(1, cdim, NPY_INT);
    int *iter = (int *) PyArray_DATA(py_iter);
    for(int i=0; i<16; i++) iter[i] = inform->checkpointsIter[i];
    PyDict_SetItemString(py_inform, "checkpointsIter", (PyObject *) py_iter);
    PyArrayObject *py_time =
      (PyArrayObject*) PyArray_SimpleNew(1, cdim, NPY_DOUBLE);
    double *time = (double *) PyArray_DATA(py_time);
    for(int i=0; i<16; i++) time[i] = inform->checkpointsTime[i];
    PyDict_SetItemString(py_inform, "checkpointsTime", (PyObject *) py_time);

    // Set time nested dictionary
    PyDict_SetItemString(py_inform, "time",
                         dqp_make_time_dict(&inform->time));

    // Set dictionaries from subservient packages
    PyDict_SetItemString(py_inform, "fdc_inform",
                         fdc_make_inform_dict(&inform->fdc_inform));
    PyDict_SetItemString(py_inform, "sls_inform",
                         sls_make_inform_dict(&inform->sls_inform));
     PyDict_SetItemString(py_inform, "sbls_inform",
                          sbls_make_inform_dict(&inform->sbls_inform));
    // PyDict_SetItemString(py_inform, "gltr_inform",
    //                      gltr_make_inform_dict(&inform->gltr_inform));
    PyDict_SetItemString(py_inform, "scu_status",
                         PyLong_FromLong(inform->scu_status));
    //PyDict_SetItemString(py_inform, "scu_inform",
    //                     scu_make_inform_dict(&inform->scu_inform));
    PyDict_SetItemString(py_inform, "rpd_inform",
                         rpd_make_inform_dict(&inform->rpd_inform));

    return py_inform;
}

//  *-*-*-*-*-*-*-*-*-*-   DQP_INITIALIZE    -*-*-*-*-*-*-*-*-*-*

static PyObject* py_dqp_initialize(PyObject *self){

    // Call dqp_initialize
    dqp_initialize(&data, &control, &status);

    // Record that DQP has been initialised
    init_called = true;

    // Return options Python dictionary
    PyObject *py_options = dqp_make_options_dict(&control);
    return Py_BuildValue("O", py_options);
}

//  *-*-*-*-*-*-*-*-*-*-*-*-   DQP_LOAD    -*-*-*-*-*-*-*-*-*-*-*-*

static PyObject* py_dqp_load(PyObject *self, PyObject *args, PyObject *keywds){
    PyArrayObject *py_H_row, *py_H_col, *py_H_ptr;
    PyArrayObject *py_A_row, *py_A_col, *py_A_ptr;
    PyObject *py_options = NULL;
    int *H_row = NULL, *H_col = NULL, *H_ptr = NULL;
    int *A_row = NULL, *A_col = NULL, *A_ptr = NULL;
    const char *A_type, *H_type;
    int n, m, A_ne, H_ne;

    // Check that package has been initialised
    if(!check_init(init_called))
        return NULL;

    // Parse positional and keyword arguments
    static char *kwlist[] = {"n","m",
                             "H_type","H_ne","H_row","H_col","H_ptr",
                             "A_type","A_ne","A_row","A_col","A_ptr",
                             "options",NULL};

    if(!PyArg_ParseTupleAndKeywords(args, keywds, "iisiOOOsiOOO|O",
                                    kwlist, &n, &m,
                                    &H_type, &H_ne, &py_H_row,
                                    &py_H_col, &py_H_ptr,
                                    &A_type, &A_ne, &py_A_row,
                                    &py_A_col, &py_A_ptr,
                                    &py_options))
        return NULL;

    // Check that array inputs are of correct type, size, and shape

    if(!(
        check_array_int("H_row", py_H_row, H_ne) &&
        check_array_int("H_col", py_H_col, H_ne) &&
        check_array_int("H_ptr", py_H_ptr, n+1)
        ))
        return NULL;
    if(!(
        check_array_int("A_row", py_A_row, A_ne) &&
        check_array_int("A_col", py_A_col, A_ne) &&
        check_array_int("A_ptr", py_A_ptr, m+1)
        ))
        return NULL;

    // Convert 64bit integer H_row array to 32bit
    if((PyObject *) py_H_row != Py_None){
        H_row = malloc(H_ne * sizeof(int));
        long int *H_row_long = (long int *) PyArray_DATA(py_H_row);
        for(int i = 0; i < H_ne; i++) H_row[i] = (int) H_row_long[i];
    }

    // Convert 64bit integer H_col array to 32bit
    if((PyObject *) py_H_col != Py_None){
        H_col = malloc(H_ne * sizeof(int));
        long int *H_col_long = (long int *) PyArray_DATA(py_H_col);
        for(int i = 0; i < H_ne; i++) H_col[i] = (int) H_col_long[i];
    }

    // Convert 64bit integer H_ptr array to 32bit
    if((PyObject *) py_H_ptr != Py_None){
        H_ptr = malloc((n+1) * sizeof(int));
        long int *H_ptr_long = (long int *) PyArray_DATA(py_H_ptr);
        for(int i = 0; i < n+1; i++) H_ptr[i] = (int) H_ptr_long[i];
    }

    // Convert 64bit integer A_row array to 32bit
    if((PyObject *) py_A_row != Py_None){
        A_row = malloc(A_ne * sizeof(int));
        long int *A_row_long = (long int *) PyArray_DATA(py_A_row);
        for(int i = 0; i < A_ne; i++) A_row[i] = (int) A_row_long[i];
    }

    // Convert 64bit integer A_col array to 32bit
    if((PyObject *) py_A_col != Py_None){
        A_col = malloc(A_ne * sizeof(int));
        long int *A_col_long = (long int *) PyArray_DATA(py_A_col);
        for(int i = 0; i < A_ne; i++) A_col[i] = (int) A_col_long[i];
    }

    // Convert 64bit integer A_ptr array to 32bit
    if((PyObject *) py_A_ptr != Py_None){
        A_ptr = malloc((m+1) * sizeof(int));
        long int *A_ptr_long = (long int *) PyArray_DATA(py_A_ptr);
        for(int i = 0; i < m+1; i++) A_ptr[i] = (int) A_ptr_long[i];
    }

    // Reset control options
    dqp_reset_control(&control, &data, &status);

    // Update DQP control options
    if(!dqp_update_control(&control, py_options))
        return NULL;

    // Call dqp_import
    dqp_import(&control, &data, &status, n, m,
               H_type, H_ne, H_row, H_col, H_ptr,
               A_type, A_ne, A_row, A_col, A_ptr);

    // Free allocated memory
    if(H_row != NULL) free(H_row);
    if(H_col != NULL) free(H_col);
    if(H_ptr != NULL) free(H_ptr);
    if(A_row != NULL) free(A_row);
    if(A_col != NULL) free(A_col);
    if(A_ptr != NULL) free(A_ptr);

    // Raise any status errors
    if(!check_error_codes(status))
        return NULL;

    // Return None boilerplate
    Py_INCREF(Py_None);
    return Py_None;
}

//  *-*-*-*-*-*-*-*-*-*-   DQP_SOLVE_QP   -*-*-*-*-*-*-*-*

static PyObject* py_dqp_solve_qp(PyObject *self, PyObject *args, PyObject *keywds){
    PyArrayObject *py_g, *py_H_val, *py_A_val;
    PyArrayObject *py_c_l, *py_c_u, *py_x_l, *py_x_u;
    PyArrayObject *py_x, *py_y, *py_z;
    double *g, *H_val, *A_val, *c_l, *c_u, *x_l, *x_u, *x, *y, *z;
    int n, m, H_ne, A_ne;
    double f;

    // Check that package has been initialised
    if(!check_init(init_called))
        return NULL;

    // Parse positional arguments
    static char *kwlist[] = {"n", "m", "f", "g", "H_ne", "H_val", "A_ne", "A_val",
                             "c_l", "c_u", "x_l", "x_u", "x", "y", "z", NULL};
    if(!PyArg_ParseTupleAndKeywords(args, keywds, "iidOiOiOOOOOOOO", kwlist, 
                                    &n, &m, &f, &py_g,
                                    &H_ne, &py_H_val, &A_ne, &py_A_val,
                                    &py_c_l, &py_c_u, &py_x_l, &py_x_u,
                                    &py_x, &py_y, &py_z))
        return NULL;

    // Check that array inputs are of correct type, size, and shape
    if(!check_array_double("g", py_g, n))
        return NULL;
    if(!check_array_double("H_val", py_H_val, H_ne))
        return NULL;
    if(!check_array_double("A_val", py_A_val, A_ne))
        return NULL;
    if(!check_array_double("c_l", py_c_l, m))
        return NULL;
    if(!check_array_double("c_u", py_c_u, m))
        return NULL;
    if(!check_array_double("x_l", py_x_l, n))
        return NULL;
    if(!check_array_double("x_u", py_x_u, n))
        return NULL;
    if(!check_array_double("x", py_x, n))
        return NULL;
    if(!check_array_double("y", py_y, m))
        return NULL;
    if(!check_array_double("z", py_z, n))
        return NULL;

    // Get array data pointer
    g = (double *) PyArray_DATA(py_g);
    H_val = (double *) PyArray_DATA(py_H_val);
    A_val = (double *) PyArray_DATA(py_A_val);
    c_l = (double *) PyArray_DATA(py_c_l);
    c_u = (double *) PyArray_DATA(py_c_u);
    x_l = (double *) PyArray_DATA(py_x_l);
    x_u = (double *) PyArray_DATA(py_x_u);
    x = (double *) PyArray_DATA(py_x);
    y = (double *) PyArray_DATA(py_y);
    z = (double *) PyArray_DATA(py_z);

    // Create NumPy output arrays
    npy_intp ndim[] = {n}; // size of x_stat
    npy_intp mdim[] = {m}; // size of c and c_ztar
    PyArrayObject *py_c =
      (PyArrayObject *) PyArray_SimpleNew(1, mdim, NPY_DOUBLE);
    double *c = (double *) PyArray_DATA(py_c);
    PyArrayObject *py_x_stat =
      (PyArrayObject *) PyArray_SimpleNew(1, ndim, NPY_INT);
    int *x_stat = (int *) PyArray_DATA(py_x_stat);
    PyArrayObject *py_c_stat =
      (PyArrayObject *) PyArray_SimpleNew(1, mdim, NPY_INT);
    int *c_stat = (int *) PyArray_DATA(py_c_stat);

    // Call dqp_solve_direct
    status = 1; // set status to 1 on entry
    dqp_solve_qp(&data, &status, n, m, H_ne, H_val, g, f, A_ne, A_val,
                 c_l, c_u, x_l, x_u, x, c, y, z, x_stat, c_stat);
    // for( int i = 0; i < n; i++) printf("x %f\n", x[i]);
    // for( int i = 0; i < m; i++) printf("c %f\n", c[i]);
    // for( int i = 0; i < n; i++) printf("x_stat %i\n", x_stat[i]);
    // for( int i = 0; i < m; i++) printf("c_stat %i\n", c_stat[i]);

    // Propagate any errors with the callback function
    if(PyErr_Occurred())
        return NULL;

    // Raise any status errors
    if(!check_error_codes(status))
        return NULL;

    // Return x, c, y, z, x_stat and c_stat
    PyObject *solve_qp_return;

    // solve_qp_return = Py_BuildValue("O", py_x);
    solve_qp_return = Py_BuildValue("OOOOOO", py_x, py_c, py_y, py_z,
                                              py_x_stat, py_c_stat);
    Py_INCREF(solve_qp_return);
    return solve_qp_return;

}
//  *-*-*-*-*-*-*-*-*-*-   DQP_SOLVE_SLDQP   -*-*-*-*-*-*-*-*

static PyObject* py_dqp_solve_sldqp(PyObject *self, PyObject *args, PyObject *keywds){
    PyArrayObject *py_g, *py_w, *py_x0, *py_A_val;
    PyArrayObject *py_c_l, *py_c_u, *py_x_l, *py_x_u;
    PyArrayObject *py_x, *py_y, *py_z;
    double *g, *w, *x0, *A_val, *c_l, *c_u, *x_l, *x_u, *x, *y, *z;
    int n, m, A_ne;
    double f;

    // Check that package has been initialised
    if(!check_init(init_called))
        return NULL;

    // Parse positional arguments
    static char *kwlist[] = {"n", "m", "f", "g", "w", "x0", "A_ne", "A_val",
                             "c_l", "c_u", "x_l", "x_u", "x", "y", "z", NULL};
    if(!PyArg_ParseTupleAndKeywords(args, keywds, "iidOOOiOOOOOOOO", kwlist, 
                                    &n, &m, &f, &py_g,
                                    &py_w, &py_x0, &A_ne, &py_A_val,
                                    &py_c_l, &py_c_u, &py_x_l, &py_x_u,
                                    &py_x, &py_y, &py_z))
        return NULL;

    // Check that array inputs are of correct type, size, and shape
    if(!check_array_double("g", py_g, n))
        return NULL;
    if(!check_array_double("w", py_w, n))
        return NULL;
    if(!check_array_double("x0", py_x0, n))
        return NULL;
    if(!check_array_double("A_val", py_A_val, A_ne))
        return NULL;
    if(!check_array_double("c_l", py_c_l, m))
        return NULL;
    if(!check_array_double("c_u", py_c_u, m))
        return NULL;
    if(!check_array_double("x_l", py_x_l, n))
        return NULL;
    if(!check_array_double("x_u", py_x_u, n))
        return NULL;
    if(!check_array_double("x", py_x, n))
        return NULL;
    if(!check_array_double("y", py_y, m))
        return NULL;
    if(!check_array_double("z", py_z, n))
        return NULL;

    // Get array data pointer
    g = (double *) PyArray_DATA(py_g);
    w = (double *) PyArray_DATA(py_w);
    x0 = (double *) PyArray_DATA(py_x0);
    A_val = (double *) PyArray_DATA(py_A_val);
    c_l = (double *) PyArray_DATA(py_c_l);
    c_u = (double *) PyArray_DATA(py_c_u);
    x_l = (double *) PyArray_DATA(py_x_l);
    x_u = (double *) PyArray_DATA(py_x_u);
    x = (double *) PyArray_DATA(py_x);
    y = (double *) PyArray_DATA(py_y);
    z = (double *) PyArray_DATA(py_z);

    // Create NumPy output arrays
    npy_intp ndim[] = {n}; // size of x_stat
    npy_intp mdim[] = {m}; // size of c and c_ztar
    PyArrayObject *py_c =
      (PyArrayObject *) PyArray_SimpleNew(1, mdim, NPY_DOUBLE);
    double *c = (double *) PyArray_DATA(py_c);
    PyArrayObject *py_x_stat =
      (PyArrayObject *) PyArray_SimpleNew(1, ndim, NPY_INT);
    int *x_stat = (int *) PyArray_DATA(py_x_stat);
    PyArrayObject *py_c_stat =
      (PyArrayObject *) PyArray_SimpleNew(1, mdim, NPY_INT);
    int *c_stat = (int *) PyArray_DATA(py_c_stat);

    // Call dqp_solve_direct
    status = 1; // set status to 1 on entry
    dqp_solve_sldqp(&data, &status, n, m, w, x0, g, f, A_ne, A_val,
                    c_l, c_u, x_l, x_u, x, c, y, z, x_stat, c_stat);

    // for( int i = 0; i < n; i++) printf("x %f\n", x[i]);
    // for( int i = 0; i < m; i++) printf("c %f\n", c[i]);
    // for( int i = 0; i < n; i++) printf("x_stat %i\n", x_stat[i]);
    // for( int i = 0; i < m; i++) printf("c_stat %i\n", c_stat[i]);

    // Propagate any errors with the callback function
    if(PyErr_Occurred())
        return NULL;

    // Raise any status errors
    if(!check_error_codes(status))
        return NULL;

    // Return x, c, y, z, x_stat and c_stat
    PyObject *solve_sldqp_return;
    solve_sldqp_return = Py_BuildValue("OOOOOO", py_x, py_c, py_y, py_z,
                                                 py_x_stat, py_c_stat);
    Py_INCREF(solve_sldqp_return);
    return solve_sldqp_return;
}

//  *-*-*-*-*-*-*-*-*-*-   DQP_INFORMATION   -*-*-*-*-*-*-*-*

static PyObject* py_dqp_information(PyObject *self){

    // Check that package has been initialised
    if(!check_init(init_called))
        return NULL;

    // Call dqp_information
    dqp_information(&data, &inform, &status);

    // Return status and inform Python dictionary
    PyObject *py_inform = dqp_make_inform_dict(&inform);
    return Py_BuildValue("O", py_inform);
}

//  *-*-*-*-*-*-*-*-*-*-   DQP_TERMINATE   -*-*-*-*-*-*-*-*-*-*

static PyObject* py_dqp_terminate(PyObject *self){

    // Check that package has been initialised
    if(!check_init(init_called))
        return NULL;

    // Call dqp_terminate
    dqp_terminate(&data, &control, &inform);

    // Return None boilerplate
    Py_INCREF(Py_None);
    return Py_None;
}

//  *-*-*-*-*-*-*-*-*-*-   INITIALIZE DQP PYTHON MODULE    -*-*-*-*-*-*-*-*-*-*

/* dqp python module method table */
static PyMethodDef dqp_module_methods[] = {
    {"initialize", (PyCFunction) py_dqp_initialize, METH_NOARGS, NULL},
    {"load", (PyCFunction) py_dqp_load, METH_VARARGS | METH_KEYWORDS, NULL},
    {"solve_qp", (PyCFunction) py_dqp_solve_qp, METH_VARARGS | METH_KEYWORDS, NULL},
    {"solve_sldqp", (PyCFunction) py_dqp_solve_sldqp, METH_VARARGS | METH_KEYWORDS, NULL},
    {"information", (PyCFunction) py_dqp_information, METH_NOARGS, NULL},
    {"terminate", (PyCFunction) py_dqp_terminate, METH_NOARGS, NULL},
    {NULL, NULL, 0, NULL}  /* Sentinel */
};

/* dqp python module documentation */

PyDoc_STRVAR(dqp_module_doc,

"The dqp package uses a dual gradient-projection method to solve the \n"
"convex quadratic programming problem to minimize the quadratic objective\n"
"q(x) = 1/2 x^T H x + g^T x + f,\n"
"or the shifted least-distance objective\n"
"1/2 sum_{j=1}^n w_j^2 ( x_j - x_j^0 )^2,\n"
"subject to the general linear constraints\n"
"c_i^l  <=  a_i^Tx  <= c_i^u, i = 1, ... , m\n"
"and the simple bound constraints\n"
"x_j^l  <=  x_j  <= x_j^u, j = 1, ... , n,\n"
"where the n by n symmetric, positive-semi-definite matrix\n"
"H, the vectors g, w, x^0,   a_i, c^l, c^u, x^l,\n"
"x^u and the scalar f are given.\n"
"Any of the constraint bounds c_i^l, c_i^u,\n"
"x_j^l and x_j^u may be infinite.\n"
"Full advantage is taken of any zero coefficients in the matrix H\n"
"or the matrix A of vectors a_i.\n"
"\n"
"See $GALAHAD/html/Python/dqp.html for argument lists, call order\n"
"and other details.\n"
"\n"
);

/* dqp python module definition */
static struct PyModuleDef module = {
   PyModuleDef_HEAD_INIT,
   "dqp",               /* name of module */
   dqp_module_doc,      /* module documentation, may be NULL */
   -1,                  /* size of per-interpreter state of the module,or -1
                           if the module keeps state in global variables */
   dqp_module_methods   /* module methods */
};

/* Python module initialization */
PyMODINIT_FUNC PyInit_dqp(void) { // must be same as module name above
    import_array();  // for NumPy arrays
    return PyModule_Create(&module);
}
