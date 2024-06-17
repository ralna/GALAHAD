//* \file clls_pyiface.c */

/*
 * THIS VERSION: GALAHAD 5.0 - 2024-06-16 AT 09:50 GMT.
 *
 *-*-*-*-*-*-*-*-*-  GALAHAD_CLLS PYTHON INTERFACE  *-*-*-*-*-*-*-*-*-*-
 *
 *  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
 *  Principal author: Jaroslav Fowkes & Nick Gould
 *
 *  History -
 *   originally released GALAHAD Version 4.1. March 20th 2023
 *
 *  For full documentation, see
 *   http://galahad.rl.ac.uk/galahad-www/specs.html
 */

#include "galahad_python.h"
#include "galahad_clls.h"

/* Nested FDC, SLS, FIT, ROOTS and CRO control and inform prototypes */
bool fdc_update_control(struct fdc_control_type *control,
                        PyObject *py_options);
PyObject* fdc_make_options_dict(const struct fdc_control_type *control);
PyObject* fdc_make_inform_dict(const struct fdc_inform_type *inform);
bool sls_update_control(struct sls_control_type *control,
                        PyObject *py_options);
PyObject* sls_make_options_dict(const struct sls_control_type *control);
PyObject* sls_make_inform_dict(const struct sls_inform_type *inform);
bool roots_update_control(struct roots_control_type *control,
                         PyObject *py_options);
PyObject* roots_make_options_dict(const struct roots_control_type *control);
PyObject* roots_make_inform_dict(const struct roots_inform_type *inform);
bool fit_update_control(struct fit_control_type *control,
                        PyObject *py_options);
PyObject* fit_make_options_dict(const struct fit_control_type *control);
PyObject* fit_make_inform_dict(const struct fit_inform_type *inform);
bool cro_update_control(struct cro_control_type *control,
                        PyObject *py_options);
PyObject* cro_make_options_dict(const struct cro_control_type *control);
PyObject* cro_make_inform_dict(const struct cro_inform_type *inform);
PyObject* rpd_make_inform_dict(const struct rpd_inform_type *inform);

/* Module global variables */
static void *data;                       // private internal data
static struct clls_control_type control;  // control struct
static struct clls_inform_type inform;    // inform struct
static bool init_called = false;         // record if initialise was called
static int status = 0;                   // exit status

//  *-*-*-*-*-*-*-*-*-*-   UPDATE CONTROL    -*-*-*-*-*-*-*-*-*-*

/* Update the control options: use C defaults but update any passed via Python*/
static bool clls_update_control(struct clls_control_type *control,
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
        if(strcmp(key_name, "maxit") == 0){
            if(!parse_int_option(value, "maxit",
                                  &control->maxit))
                return false;
            continue;
        }
        if(strcmp(key_name, "infeas_max") == 0){
            if(!parse_int_option(value, "infeas_max",
                                  &control->infeas_max))
                return false;
            continue;
        }
        if(strcmp(key_name, "muzero_fixed") == 0){
            if(!parse_int_option(value, "muzero_fixed",
                                  &control->muzero_fixed))
                return false;
            continue;
        }
        if(strcmp(key_name, "restore_problem") == 0){
            if(!parse_int_option(value, "restore_problem",
                                  &control->restore_problem))
                return false;
            continue;
        }
        if(strcmp(key_name, "indicator_type") == 0){
            if(!parse_int_option(value, "indicator_type",
                                  &control->indicator_type))
                return false;
            continue;
        }
        if(strcmp(key_name, "arc") == 0){
            if(!parse_int_option(value, "arc",
                                  &control->arc))
                return false;
            continue;
        }
        if(strcmp(key_name, "series_order") == 0){
            if(!parse_int_option(value, "series_order",
                                  &control->series_order))
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
        if(strcmp(key_name, "prfeas") == 0){
            if(!parse_double_option(value, "prfeas",
                                  &control->prfeas))
                return false;
            continue;
        }
        if(strcmp(key_name, "dufeas") == 0){
            if(!parse_double_option(value, "dufeas",
                                  &control->dufeas))
                return false;
            continue;
        }
        if(strcmp(key_name, "muzero") == 0){
            if(!parse_double_option(value, "muzero",
                                  &control->muzero))
                return false;
            continue;
        }
        if(strcmp(key_name, "tau") == 0){
            if(!parse_double_option(value, "tau",
                                  &control->tau))
                return false;
            continue;
        }
        if(strcmp(key_name, "gamma_c") == 0){
            if(!parse_double_option(value, "gamma_c",
                                  &control->gamma_c))
                return false;
            continue;
        }
        if(strcmp(key_name, "gamma_f") == 0){
            if(!parse_double_option(value, "gamma_f",
                                  &control->gamma_f))
                return false;
            continue;
        }
        if(strcmp(key_name, "reduce_infeas") == 0){
            if(!parse_double_option(value, "reduce_infeas",
                                  &control->reduce_infeas))
                return false;
            continue;
        }
        if(strcmp(key_name, "identical_bounds_tol") == 0){
            if(!parse_double_option(value, "identical_bounds_tol",
                                  &control->identical_bounds_tol))
                return false;
            continue;
        }
        if(strcmp(key_name, "mu_pounce") == 0){
            if(!parse_double_option(value, "mu_pounce",
                                  &control->mu_pounce))
                return false;
            continue;
        }
        if(strcmp(key_name, "indicator_tol_p") == 0){
            if(!parse_double_option(value, "indicator_tol_p",
                                  &control->indicator_tol_p))
                return false;
            continue;
        }
        if(strcmp(key_name, "indicator_tol_pd") == 0){
            if(!parse_double_option(value, "indicator_tol_pd",
                                  &control->indicator_tol_pd))
                return false;
            continue;
        }
        if(strcmp(key_name, "indicator_tol_tapia") == 0){
            if(!parse_double_option(value, "indicator_tol_tapia",
                                  &control->indicator_tol_tapia))
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
        if(strcmp(key_name, "treat_separable_as_general") == 0){
            if(!parse_bool_option(value, "treat_separable_as_general",
                                  &control->treat_separable_as_general))
                return false;
            continue;
        }
        if(strcmp(key_name, "just_feasible") == 0){
            if(!parse_bool_option(value, "just_feasible",
                                  &control->just_feasible))
                return false;
            continue;
        }
        if(strcmp(key_name, "getdua") == 0){
            if(!parse_bool_option(value, "getdua",
                                  &control->getdua))
                return false;
            continue;
        }
        if(strcmp(key_name, "puiseux") == 0){
            if(!parse_bool_option(value, "puiseux",
                                  &control->puiseux))
                return false;
            continue;
        }
        if(strcmp(key_name, "every_order") == 0){
            if(!parse_bool_option(value, "every_order",
                                  &control->every_order))
                return false;
            continue;
        }
        if(strcmp(key_name, "feasol") == 0){
            if(!parse_bool_option(value, "feasol",
                                  &control->feasol))
                return false;
            continue;
        }
        if(strcmp(key_name, "balance_initial_complentarity") == 0){
            if(!parse_bool_option(value, "balance_initial_complentarity",
                                  &control->balance_initial_complentarity))
                return false;
            continue;
        }
        if(strcmp(key_name, "crossover") == 0){
            if(!parse_bool_option(value, "crossover",
                                  &control->crossover))
                return false;
            continue;
        }
        if(strcmp(key_name, "reduced_pounce_system") == 0){
            if(!parse_bool_option(value, "reduced_pounce_system",
                                  &control->reduced_pounce_system))
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
        if(strcmp(key_name, "sls_pounce_options") == 0){
            if(!sls_update_control(&control->sls_pounce_control, value))
                return false;
            continue;
        }
        if(strcmp(key_name, "fit_options") == 0){
            if(!fit_update_control(&control->fit_control, value))
                return false;
            continue;
        }
        if(strcmp(key_name, "roots_options") == 0){
            if(!roots_update_control(&control->roots_control, value))
                return false;
            continue;
        }
        if(strcmp(key_name, "cro_options") == 0){
            if(!cro_update_control(&control->cro_control, value))
                return false;
            continue;
        }

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
PyObject* clls_make_options_dict(const struct clls_control_type *control){
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
    PyDict_SetItemString(py_options, "maxit",
                         PyLong_FromLong(control->maxit));
    PyDict_SetItemString(py_options, "infeas_max",
                         PyLong_FromLong(control->infeas_max));
    PyDict_SetItemString(py_options, "muzero_fixed",
                         PyLong_FromLong(control->muzero_fixed));
    PyDict_SetItemString(py_options, "restore_problem",
                         PyLong_FromLong(control->restore_problem));
    PyDict_SetItemString(py_options, "indicator_type",
                         PyLong_FromLong(control->indicator_type));
    PyDict_SetItemString(py_options, "arc",
                         PyLong_FromLong(control->arc));
    PyDict_SetItemString(py_options, "series_order",
                         PyLong_FromLong(control->series_order));
    PyDict_SetItemString(py_options, "sif_file_device",
                         PyLong_FromLong(control->sif_file_device));
    PyDict_SetItemString(py_options, "qplib_file_device",
                         PyLong_FromLong(control->qplib_file_device));
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
    PyDict_SetItemString(py_options, "prfeas",
                         PyFloat_FromDouble(control->prfeas));
    PyDict_SetItemString(py_options, "dufeas",
                         PyFloat_FromDouble(control->dufeas));
    PyDict_SetItemString(py_options, "muzero",
                         PyFloat_FromDouble(control->muzero));
    PyDict_SetItemString(py_options, "tau",
                         PyFloat_FromDouble(control->tau));
    PyDict_SetItemString(py_options, "gamma_c",
                         PyFloat_FromDouble(control->gamma_c));
    PyDict_SetItemString(py_options, "gamma_f",
                         PyFloat_FromDouble(control->gamma_f));
    PyDict_SetItemString(py_options, "reduce_infeas",
                         PyFloat_FromDouble(control->reduce_infeas));
    PyDict_SetItemString(py_options, "identical_bounds_tol",
                         PyFloat_FromDouble(control->identical_bounds_tol));
    PyDict_SetItemString(py_options, "mu_pounce",
                         PyFloat_FromDouble(control->mu_pounce));
    PyDict_SetItemString(py_options, "indicator_tol_p",
                         PyFloat_FromDouble(control->indicator_tol_p));
    PyDict_SetItemString(py_options, "indicator_tol_pd",
                         PyFloat_FromDouble(control->indicator_tol_pd));
    PyDict_SetItemString(py_options, "indicator_tol_tapia",
                         PyFloat_FromDouble(control->indicator_tol_tapia));
    PyDict_SetItemString(py_options, "cpu_time_limit",
                         PyFloat_FromDouble(control->cpu_time_limit));
    PyDict_SetItemString(py_options, "clock_time_limit",
                         PyFloat_FromDouble(control->clock_time_limit));
    PyDict_SetItemString(py_options, "remove_dependencies",
                         PyBool_FromLong(control->remove_dependencies));
    PyDict_SetItemString(py_options, "treat_zero_bounds_as_general",
                         PyBool_FromLong(control->treat_zero_bounds_as_general));
    PyDict_SetItemString(py_options, "treat_separable_as_general",
                         PyBool_FromLong(control->treat_separable_as_general));
    PyDict_SetItemString(py_options, "just_feasible",
                         PyBool_FromLong(control->just_feasible));
    PyDict_SetItemString(py_options, "getdua",
                         PyBool_FromLong(control->getdua));
    PyDict_SetItemString(py_options, "puiseux",
                         PyBool_FromLong(control->puiseux));
    PyDict_SetItemString(py_options, "every_order",
                         PyBool_FromLong(control->every_order));
    PyDict_SetItemString(py_options, "feasol",
                         PyBool_FromLong(control->feasol));
    PyDict_SetItemString(py_options, "balance_initial_complentarity",
                         PyBool_FromLong(control->balance_initial_complentarity));
    PyDict_SetItemString(py_options, "crossover",
                         PyBool_FromLong(control->crossover));
    PyDict_SetItemString(py_options, "reduced_pounce_system",
                         PyBool_FromLong(control->reduced_pounce_system));
    PyDict_SetItemString(py_options, "space_critical",
                         PyBool_FromLong(control->space_critical));
    PyDict_SetItemString(py_options, "deallocate_error_fatal",
                         PyBool_FromLong(control->deallocate_error_fatal));
    PyDict_SetItemString(py_options, "generate_sif_file",
                         PyBool_FromLong(control->generate_sif_file));
    PyDict_SetItemString(py_options, "generate_qplib_file",
                         PyBool_FromLong(control->generate_qplib_file));
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
    PyDict_SetItemString(py_options, "sls_pounce_options",
                         sls_make_options_dict(&control->sls_pounce_control));
    PyDict_SetItemString(py_options, "fit_options",
                         fit_make_options_dict(&control->fit_control));
    PyDict_SetItemString(py_options, "roots_options",
                         roots_make_options_dict(&control->roots_control));
    PyDict_SetItemString(py_options, "cro_options",
                         cro_make_options_dict(&control->cro_control));

    return py_options;
}


//  *-*-*-*-*-*-*-*-*-*-   MAKE TIME    -*-*-*-*-*-*-*-*-*-*

/* Take the time struct from C and turn it into a python dictionary */
static PyObject* clls_make_time_dict(const struct clls_time_type *time){
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

    return py_time;
}

//  *-*-*-*-*-*-*-*-*-*-   MAKE INFORM    -*-*-*-*-*-*-*-*-*-*

/* Take the inform struct from C and turn it into a python dictionary */
static PyObject* clls_make_inform_dict(const struct clls_inform_type *inform){
    PyObject *py_inform = PyDict_New();

    PyDict_SetItemString(py_inform, "status",
                         PyLong_FromLong(inform->status));
    PyDict_SetItemString(py_inform, "alloc_status",
                         PyLong_FromLong(inform->alloc_status));
    PyDict_SetItemString(py_inform, "bad_alloc",
                         PyUnicode_FromString(inform->bad_alloc));
    PyDict_SetItemString(py_inform, "iter",
                         PyLong_FromLong(inform->iter));
    PyDict_SetItemString(py_inform, "factorization_status",
                         PyLong_FromLong(inform->factorization_status));
    PyDict_SetItemString(py_inform, "factorization_integer",
                         PyLong_FromLong(inform->factorization_integer));
    PyDict_SetItemString(py_inform, "factorization_real",
                         PyLong_FromLong(inform->factorization_real));
    PyDict_SetItemString(py_inform, "nfacts",
                         PyLong_FromLong(inform->nfacts));
    PyDict_SetItemString(py_inform, "nbacts",
                         PyLong_FromLong(inform->nbacts));
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
                         clls_make_time_dict(&inform->time));

    // Set dictionaries from subservient packages
    PyDict_SetItemString(py_inform, "fdc_inform",
                         fdc_make_inform_dict(&inform->fdc_inform));
    PyDict_SetItemString(py_inform, "sls_inform",
                        sls_make_inform_dict(&inform->sls_inform));
    PyDict_SetItemString(py_inform, "sls_pounce_inform",
                        sls_make_inform_dict(&inform->sls_pounce_inform));
    PyDict_SetItemString(py_inform, "fit_inform",
                         fit_make_inform_dict(&inform->fit_inform));
    PyDict_SetItemString(py_inform, "roots_inform",
                         roots_make_inform_dict(&inform->roots_inform));
    PyDict_SetItemString(py_inform, "cro_inform",
                         cro_make_inform_dict(&inform->cro_inform));
    PyDict_SetItemString(py_inform, "rpd_inform",
                         rpd_make_inform_dict(&inform->rpd_inform));

    return py_inform;
}

//  *-*-*-*-*-*-*-*-*-*-   CLLS_INITIALIZE    -*-*-*-*-*-*-*-*-*-*

static PyObject* py_clls_initialize(PyObject *self){

    // Call clls_initialize
    clls_initialize(&data, &control, &status);

    // Record that CLLS has been initialised
    init_called = true;

    // Return options Python dictionary
    PyObject *py_options = clls_make_options_dict(&control);
    return Py_BuildValue("O", py_options);
}

//  *-*-*-*-*-*-*-*-*-*-*-*-   CLLS_LOAD    -*-*-*-*-*-*-*-*-*-*-*-*

static PyObject* py_clls_load(PyObject *self, PyObject *args, PyObject *keywds){
    PyArrayObject *py_Ao_row, *py_Ao_col, *py_Ao_ptr;
    PyArrayObject *py_A_row, *py_A_col, *py_A_ptr;
    PyObject *py_options = NULL;
    int *Ao_row = NULL, *Ao_col = NULL, *Ao_ptr = NULL;
    int *A_row = NULL, *A_col = NULL, *A_ptr = NULL;
    const char *Ao_type, *A_type;
    int n, o, m, Ao_ne, Ao_ptr_ne, A_ne, A_ptr_ne;

    // Check that package has been initialised
    if(!check_init(init_called))
        return NULL;

    // Parse positional and keyword arguments
    static char *kwlist[] = {"n","o","m",
                             "Ao_type","Ao_ne","Ao_row",
                             "Ao_col","Ao_ptr_ne","Ao_ptr",
                             "A_type","A_ne","A_row",
                             "A_col","A_ptr_ne","A_ptr",
                             "options",NULL};

    if(!PyArg_ParseTupleAndKeywords(args, keywds, "iiisiOOiOsiOOiO|O",
                                    kwlist, &n, &o, &m,
                                    &Ao_type, &Ao_ne, &py_Ao_row,
                                    &py_Ao_col, &Ao_ptr_ne, &py_Ao_ptr,
                                    &A_type, &A_ne, &py_A_row,
                                    &py_A_col, &A_ptr_ne, &py_A_ptr,
                                    &py_options))
        return NULL;

    // Check that array inputs are of correct type, size, and shape

    if(!(
        check_array_int("Ao_row", py_Ao_row, Ao_ne) &&
        check_array_int("Ao_col", py_Ao_col, Ao_ne) &&
        check_array_int("Ao_ptr", py_Ao_ptr, Ao_ptr_ne)
        ))
        return NULL;
    if(!(
        check_array_int("A_row", py_A_row, A_ne) &&
        check_array_int("A_col", py_A_col, A_ne) &&
        check_array_int("A_ptr", py_A_ptr, A_ptr_ne)
        ))
        return NULL;

    // Convert 64bit integer Ao_row array to 32bit
    if((PyObject *) py_Ao_row != Py_None){
        Ao_row = malloc(Ao_ne * sizeof(int));
        long int *Ao_row_long = (long int *) PyArray_DATA(py_Ao_row);
        for(int i = 0; i < Ao_ne; i++) Ao_row[i] = (int) Ao_row_long[i];
    }

    // Convert 64bit integer Ao_col array to 32bit
    if((PyObject *) py_Ao_col != Py_None){
        Ao_col = malloc(Ao_ne * sizeof(int));
        long int *Ao_col_long = (long int *) PyArray_DATA(py_Ao_col);
        for(int i = 0; i < Ao_ne; i++) Ao_col[i] = (int) Ao_col_long[i];
    }

    // Convert 64bit integer Ao_ptr array to 32bit
    if((PyObject *) py_Ao_ptr != Py_None){
        Ao_ptr = malloc((Ao_ptr_ne) * sizeof(int));
        long int *Ao_ptr_long = (long int *) PyArray_DATA(py_Ao_ptr);
        for(int i = 0; i < Ao_ptr_ne; i++) Ao_ptr[i] = (int) Ao_ptr_long[i];
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
        A_ptr = malloc((A_ptr_ne) * sizeof(int));
        long int *A_ptr_long = (long int *) PyArray_DATA(py_A_ptr);
        for(int i = 0; i < A_ptr_ne; i++) A_ptr[i] = (int) A_ptr_long[i];
    }

    // Reset control options
    clls_reset_control(&control, &data, &status);

    // Update CLLS control options
    if(!clls_update_control(&control, py_options))
        return NULL;

    // Call clls_import
    clls_import(&control, &data, &status, n, o, m,
                Ao_type, Ao_ne, Ao_row, Ao_col, Ao_ptr_ne, Ao_ptr,
                A_type, A_ne, A_row, A_col, A_ptr_ne, A_ptr );

    // Free allocated memory
    if(Ao_row != NULL) free(Ao_row);
    if(Ao_col != NULL) free(Ao_col);
    if(Ao_ptr != NULL) free(Ao_ptr);
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

//  *-*-*-*-*-*-*-*-*-*-   CLLS_SOLVE_CLLS   -*-*-*-*-*-*-*-*

static PyObject* py_clls_solve_clls(PyObject *self, PyObject *args, PyObject *keywds){
    PyArrayObject *py_Ao_val, *py_b, *py_A_val;
    PyArrayObject *py_c_l, *py_c_u, *py_x_l, *py_x_u, *py_w;
    PyArrayObject *py_x, *py_y, *py_z;
    double *Ao_val, *b, *A_val, *c_l, *c_u, *x_l, *x_u, *w, *x, *y, *z;
    int n, o, m, Ao_ne, A_ne;
    double sigma;

    // Check that package has been initialised
    if(!check_init(init_called))
        return NULL;

    // Parse positional and keyword arguments
    static char *kwlist[] = {"n","o","m", "Ao_ne","Ao_val", "b", "sigma", 
                             "A_ne","A_val", "c_l", "c_u", "x_l", "x_u", 
                             "x", "y", "z", "w", NULL};

    if(!PyArg_ParseTupleAndKeywords(args, keywds, "iiiiOOdiOOOOOOOOO",
                                    kwlist, &n, &o, &m,
                                    &Ao_ne, &py_Ao_val, &py_b, &sigma, 
                                    &A_ne, &py_A_val,
                                    &py_c_l, &py_c_u, &py_x_l, &py_x_u,
                                    &py_x, &py_y, &py_z, &py_w))
        return NULL;

    // Check that array inputs are of correct type, size, and shape
    if(!check_array_double("Ao_val", py_Ao_val, Ao_ne))
        return NULL;
    if(!check_array_double("b", py_b, o))
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
    if(!check_array_double("w", py_w, o))
        return NULL;

    // Get array data pointer
    Ao_val = (double *) PyArray_DATA(py_Ao_val);
    b = (double *) PyArray_DATA(py_b);
    A_val = (double *) PyArray_DATA(py_A_val);
    c_l = (double *) PyArray_DATA(py_c_l);
    c_u = (double *) PyArray_DATA(py_c_u);
    x_l = (double *) PyArray_DATA(py_x_l);
    x_u = (double *) PyArray_DATA(py_x_u);
    x = (double *) PyArray_DATA(py_x);
    y = (double *) PyArray_DATA(py_y);
    z = (double *) PyArray_DATA(py_z);
    w = (double *) PyArray_DATA(py_w);

   // Create NumPy output arrays
    npy_intp ndim[] = {n}; // size of x_stat
    npy_intp odim[] = {o}; // size of r
    npy_intp mdim[] = {m}; // size of c and c_stat
    PyArrayObject *py_r = 
      (PyArrayObject *) PyArray_SimpleNew(1, odim, NPY_DOUBLE);
    double *r = (double *) PyArray_DATA(py_r);
    PyArrayObject *py_c = 
      (PyArrayObject *) PyArray_SimpleNew(1, mdim, NPY_DOUBLE);
    double *c = (double *) PyArray_DATA(py_c);
    PyArrayObject *py_x_stat = 
      (PyArrayObject *) PyArray_SimpleNew(1, ndim, NPY_INT);
    int *x_stat = (int *) PyArray_DATA(py_x_stat);
    PyArrayObject *py_c_stat = 
      (PyArrayObject *) PyArray_SimpleNew(1, mdim, NPY_INT);
    int *c_stat = (int *) PyArray_DATA(py_c_stat);

    // Call clls_solve_direct
    status = 1; // set status to 1 on entry
    clls_solve_clls(&data, &status, n, o, m, 
                    Ao_ne, Ao_val, b, sigma, A_ne, A_val, 
                    c_l, c_u, x_l, x_u, x, r, c, y, z, x_stat, c_stat, w);
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

    // Return x, r, c, y, z, x_stat and c_stat
    PyObject *solve_clls_return;

    // solve_qp_return = Py_BuildValue("O", py_x);
    solve_clls_return = Py_BuildValue("OOOOOOO", py_x, py_r, py_c, py_y, py_z, 
                                                 py_x_stat, py_c_stat);
    Py_INCREF(solve_clls_return);
    return solve_clls_return;

}

//  *-*-*-*-*-*-*-*-*-*-   CLLS_INFORMATION   -*-*-*-*-*-*-*-*

static PyObject* py_clls_information(PyObject *self){

    // Check that package has been initialised
    if(!check_init(init_called))
        return NULL;

    // Call clls_information
    clls_information(&data, &inform, &status);

    // Return status and inform Python dictionary
    PyObject *py_inform = clls_make_inform_dict(&inform);
    return Py_BuildValue("O", py_inform);
}

//  *-*-*-*-*-*-*-*-*-*-   CLLS_TERMINATE   -*-*-*-*-*-*-*-*-*-*

static PyObject* py_clls_terminate(PyObject *self){

    // Check that package has been initialised
    if(!check_init(init_called))
        return NULL;

    // Call clls_terminate
    clls_terminate(&data, &control, &inform);

    // Return None boilerplate
    Py_INCREF(Py_None);
    return Py_None;
}

//  *-*-*-*-*-*-*-*-*-*-   INITIALIZE CLLS PYTHON MODULE    -*-*-*-*-*-*-*-*-*-*

/* clls python module method table */
static PyMethodDef clls_module_methods[] = {
    {"initialize", (PyCFunction) py_clls_initialize, METH_NOARGS, NULL},
    {"load", (PyCFunction) py_clls_load, METH_VARARGS | METH_KEYWORDS, NULL},
    {"solve_clls", (PyCFunction) py_clls_solve_clls, METH_VARARGS | METH_KEYWORDS, NULL},
    {"information", (PyCFunction) py_clls_information, METH_NOARGS, NULL},
    {"terminate", (PyCFunction) py_clls_terminate, METH_NOARGS, NULL},
    {NULL, NULL, 0, NULL}  /* Sentinel */
};

/* clls python module documentation */

PyDoc_STRVAR(clls_module_doc,

"The clls package uses a primal-dual interior-point method to solve the\n"
"convex quadratic programming problem to minimize the regularized linear\n"
"least-squares objective\n"
"q(x) = 1/2 ||Ao x - b||_W^2 + sigma/2 ||x||^2,\n"
"subject to the general linear constraints\n"
"c_i^l  <=  a_i^Tx  <= c_i^u, i = 1, ... , m\n"
"and the simple bound constraints\n"
"x_j^l  <=  x_j  <= x_j^u, j = 1, ... , n,\n"
"where the o by n matrix Ao, the vectors b, w, a_i, c^l, c^u," 
"x^l, x^u and the regularization weight sigma >= 0 are given,\n"
"and the norms are defined by ||v||_W^2 = v^T W v and ||v||^2 = v^T v,\n"
"where W is the digonal matrix whose entries are the components of w > 0\n"
"Any of the constraint bounds c_i^l, c_i^u,\n"
"x_j^l and x_j^u may be infinite.\n"
"Full advantage is taken of any zero coefficients in the matrix Ao\n"
"or the matrix A of vectors a_i.\n"
"\n"
"See $GALAHAD/html/Python/clls.html for argument lists, call order\n"
"and other details.\n"
"\n"
);

/* clls python module definition */
static struct PyModuleDef module = {
   PyModuleDef_HEAD_INIT,
   "clls",               /* name of module */
   clls_module_doc,      /* module documentation, may be NULL */
   -1,                  /* size of per-interpreter state of the module,or -1
                           if the module keeps state in global variables */
   clls_module_methods   /* module methods */
};

/* Python module initialization */
PyMODINIT_FUNC PyInit_clls(void) { // must be same as module name above
    import_array();  // for NumPy arrays
    return PyModule_Create(&module);
}
