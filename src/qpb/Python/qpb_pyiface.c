//* \file qpb_pyiface.c */

/*
 * THIS VERSION: GALAHAD 4.1 - 2023-05-20 AT 10:30 GMT.
 *
 *-*-*-*-*-*-*-*-*-  GALAHAD_QPB PYTHON INTERFACE  *-*-*-*-*-*-*-*-*-*-
 *
 *  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
 *  Principal author: Jaroslav Fowkes & Nick Gould
 *
 *  History -
 *   originally released GALAHAD Version 4.1. April 5th 2023
 *
 *  For full documentation, see
 *   http://galahad.rl.ac.uk/galahad-www/specs.html
 */

#include "galahad_python.h"
#include "galahad_qpb.h"

/* Nested LSQP, FDC, SBLS, GLTR and FDC control and inform prototypes */
bool lsqp_update_control(struct lsqp_control_type *control,
                         PyObject *py_options);
PyObject* lsqp_make_options_dict(const struct lsqp_control_type *control);
PyObject* lsqp_make_inform_dict(const struct lsqp_inform_type *inform);
bool fdc_update_control(struct fdc_control_type *control,
                        PyObject *py_options);
PyObject* fdc_make_options_dict(const struct fdc_control_type *control);
PyObject* fdc_make_inform_dict(const struct fdc_inform_type *inform);
bool sbls_update_control(struct sbls_control_type *control,
                         PyObject *py_options);
PyObject* sbls_make_options_dict(const struct sbls_control_type *control);
PyObject* sbls_make_inform_dict(const struct sbls_inform_type *inform);
bool gltr_update_control(struct gltr_control_type *control,
                         PyObject *py_options);
PyObject* gltr_make_options_dict(const struct gltr_control_type *control);
PyObject* gltr_make_inform_dict(const struct gltr_inform_type *inform);
bool fit_update_control(struct fit_control_type *control,
                        PyObject *py_options);
PyObject* fit_make_options_dict(const struct fit_control_type *control);
PyObject* fit_make_inform_dict(const struct fit_inform_type *inform);

/* Module global variables */
static void *data;                       // private internal data
static struct qpb_control_type control;  // control struct
static struct qpb_inform_type inform;    // inform struct
static bool init_called = false;         // record if initialise was called
static int status = 0;                   // exit status

//  *-*-*-*-*-*-*-*-*-*-   UPDATE CONTROL    -*-*-*-*-*-*-*-*-*-*

/* Update the control options: use C defaults but update any passed via Python*/
static bool qpb_update_control(struct qpb_control_type *control,
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
        if(strcmp(key_name, "itref_max") == 0){
            if(!parse_int_option(value, "itref_max",
                                  &control->itref_max))
                return false;
            continue;
        }
        if(strcmp(key_name, "cg_maxit") == 0){
            if(!parse_int_option(value, "cg_maxit",
                                  &control->cg_maxit))
                return false;
            continue;
        }
        if(strcmp(key_name, "indicator_type") == 0){
            if(!parse_int_option(value, "indicator_type",
                                  &control->indicator_type))
                return false;
            continue;
        }
        if(strcmp(key_name, "restore_problem") == 0){
            if(!parse_int_option(value, "restore_problem",
                                  &control->restore_problem))
                return false;
            continue;
        }
        if(strcmp(key_name, "extrapolate") == 0){
            if(!parse_int_option(value, "extrapolate",
                                  &control->extrapolate))
                return false;
            continue;
        }
        if(strcmp(key_name, "path_history") == 0){
            if(!parse_int_option(value, "path_history",
                                  &control->path_history))
                return false;
            continue;
        }
        if(strcmp(key_name, "factor") == 0){
            if(!parse_int_option(value, "factor",
                                  &control->factor))
                return false;
            continue;
        }
        if(strcmp(key_name, "max_col") == 0){
            if(!parse_int_option(value, "max_col",
                                  &control->max_col))
                return false;
            continue;
        }
        if(strcmp(key_name, "indmin") == 0){
            if(!parse_int_option(value, "indmin",
                                  &control->indmin))
                return false;
            continue;
        }
        if(strcmp(key_name, "valmin") == 0){
            if(!parse_int_option(value, "valmin",
                                  &control->valmin))
                return false;
            continue;
        }
        if(strcmp(key_name, "infeas_max") == 0){
            if(!parse_int_option(value, "infeas_max",
                                  &control->infeas_max))
                return false;
            continue;
        }
        if(strcmp(key_name, "precon") == 0){
            if(!parse_int_option(value, "precon",
                                  &control->precon))
                return false;
            continue;
        }
        if(strcmp(key_name, "nsemib") == 0){
            if(!parse_int_option(value, "nsemib",
                                  &control->nsemib))
                return false;
            continue;
        }
        if(strcmp(key_name, "path_derivatives") == 0){
            if(!parse_int_option(value, "path_derivatives",
                                  &control->path_derivatives))
                return false;
            continue;
        }
        if(strcmp(key_name, "fit_order") == 0){
            if(!parse_int_option(value, "fit_order",
                                  &control->fit_order))
                return false;
            continue;
        }
        if(strcmp(key_name, "sif_file_device") == 0){
            if(!parse_int_option(value, "sif_file_device",
                                  &control->sif_file_device))
                return false;
            continue;
        }
        if(strcmp(key_name, "infinity") == 0){
            if(!parse_double_option(value, "infinity",
                                  &control->infinity))
                return false;
            continue;
        }
        if(strcmp(key_name, "stop_p") == 0){
            if(!parse_double_option(value, "stop_p",
                                  &control->stop_p))
                return false;
            continue;
        }
        if(strcmp(key_name, "stop_d") == 0){
            if(!parse_double_option(value, "stop_d",
                                  &control->stop_d))
                return false;
            continue;
        }
        if(strcmp(key_name, "stop_c") == 0){
            if(!parse_double_option(value, "stop_c",
                                  &control->stop_c))
                return false;
            continue;
        }
        if(strcmp(key_name, "theta_d") == 0){
            if(!parse_double_option(value, "theta_d",
                                  &control->theta_d))
                return false;
            continue;
        }
        if(strcmp(key_name, "theta_c") == 0){
            if(!parse_double_option(value, "theta_c",
                                  &control->theta_c))
                return false;
            continue;
        }
        if(strcmp(key_name, "beta") == 0){
            if(!parse_double_option(value, "beta",
                                  &control->beta))
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
        if(strcmp(key_name, "reduce_infeas") == 0){
            if(!parse_double_option(value, "reduce_infeas",
                                  &control->reduce_infeas))
                return false;
            continue;
        }
        if(strcmp(key_name, "obj_unbounded") == 0){
            if(!parse_double_option(value, "obj_unbounded",
                                  &control->obj_unbounded))
                return false;
            continue;
        }
        if(strcmp(key_name, "pivot_tol") == 0){
            if(!parse_double_option(value, "pivot_tol",
                                  &control->pivot_tol))
                return false;
            continue;
        }
        if(strcmp(key_name, "pivot_tol_for_dependencies") == 0){
            if(!parse_double_option(value, "pivot_tol_for_dependencies",
                                  &control->pivot_tol_for_dependencies))
                return false;
            continue;
        }
        if(strcmp(key_name, "zero_pivot") == 0){
            if(!parse_double_option(value, "zero_pivot",
                                  &control->zero_pivot))
                return false;
            continue;
        }
        if(strcmp(key_name, "identical_bounds_tol") == 0){
            if(!parse_double_option(value, "identical_bounds_tol",
                                  &control->identical_bounds_tol))
                return false;
            continue;
        }
        if(strcmp(key_name, "inner_stop_relative") == 0){
            if(!parse_double_option(value, "inner_stop_relative",
                                  &control->inner_stop_relative))
                return false;
            continue;
        }
        if(strcmp(key_name, "inner_stop_absolute") == 0){
            if(!parse_double_option(value, "inner_stop_absolute",
                                  &control->inner_stop_absolute))
                return false;
            continue;
        }
        if(strcmp(key_name, "initial_radius") == 0){
            if(!parse_double_option(value, "initial_radius",
                                  &control->initial_radius))
                return false;
            continue;
        }
        if(strcmp(key_name, "mu_min") == 0){
            if(!parse_double_option(value, "mu_min",
                                  &control->mu_min))
                return false;
            continue;
        }
        if(strcmp(key_name, "inner_fraction_opt") == 0){
            if(!parse_double_option(value, "inner_fraction_opt",
                                  &control->inner_fraction_opt))
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
        if(strcmp(key_name, "center") == 0){
            if(!parse_bool_option(value, "center",
                                  &control->center))
                return false;
            continue;
        }
        if(strcmp(key_name, "primal") == 0){
            if(!parse_bool_option(value, "primal",
                                  &control->primal))
                return false;
            continue;
        }
        if(strcmp(key_name, "puiseux") == 0){
            if(!parse_bool_option(value, "puiseux",
                                  &control->puiseux))
                return false;
            continue;
        }
        if(strcmp(key_name, "feasol") == 0){
            if(!parse_bool_option(value, "feasol",
                                  &control->feasol))
                return false;
            continue;
        }
        if(strcmp(key_name, "array_syntax_worse_than_do_loop") == 0){
            if(!parse_bool_option(value, "array_syntax_worse_than_do_loop",
                                  &control->array_syntax_worse_than_do_loop))
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
        if(strcmp(key_name, "sif_file_name") == 0){
            if(!parse_char_option(value, "sif_file_name",
                                  control->sif_file_name,
                                  sizeof(control->sif_file_name)))
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
        if(strcmp(key_name, "lsqp_options") == 0){
            if(!lsqp_update_control(&control->lsqp_control, value))
                return false;
            continue;
        }
        if(strcmp(key_name, "fdc_options") == 0){
            if(!fdc_update_control(&control->fdc_control, value))
                return false;
            continue;
        }
        if(strcmp(key_name, "sbls_options") == 0){
            if(!sbls_update_control(&control->sbls_control, value))
                return false;
            continue;
        }
        if(strcmp(key_name, "gltr_options") == 0){
            if(!gltr_update_control(&control->gltr_control, value))
                return false;
            continue;
        }
        if(strcmp(key_name, "fit_options") == 0){
            if(!fit_update_control(&control->fit_control, value))
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
PyObject* qpb_make_options_dict(const struct qpb_control_type *control){
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
    PyDict_SetItemString(py_options, "itref_max",
                         PyLong_FromLong(control->itref_max));
    PyDict_SetItemString(py_options, "cg_maxit",
                         PyLong_FromLong(control->cg_maxit));
    PyDict_SetItemString(py_options, "indicator_type",
                         PyLong_FromLong(control->indicator_type));
    PyDict_SetItemString(py_options, "restore_problem",
                         PyLong_FromLong(control->restore_problem));
    PyDict_SetItemString(py_options, "extrapolate",
                         PyLong_FromLong(control->extrapolate));
    PyDict_SetItemString(py_options, "path_history",
                         PyLong_FromLong(control->path_history));
    PyDict_SetItemString(py_options, "factor",
                         PyLong_FromLong(control->factor));
    PyDict_SetItemString(py_options, "max_col",
                         PyLong_FromLong(control->max_col));
    PyDict_SetItemString(py_options, "indmin",
                         PyLong_FromLong(control->indmin));
    PyDict_SetItemString(py_options, "valmin",
                         PyLong_FromLong(control->valmin));
    PyDict_SetItemString(py_options, "infeas_max",
                         PyLong_FromLong(control->infeas_max));
    PyDict_SetItemString(py_options, "precon",
                         PyLong_FromLong(control->precon));
    PyDict_SetItemString(py_options, "nsemib",
                         PyLong_FromLong(control->nsemib));
    PyDict_SetItemString(py_options, "path_derivatives",
                         PyLong_FromLong(control->path_derivatives));
    PyDict_SetItemString(py_options, "fit_order",
                         PyLong_FromLong(control->fit_order));
    PyDict_SetItemString(py_options, "sif_file_device",
                         PyLong_FromLong(control->sif_file_device));
    PyDict_SetItemString(py_options, "infinity",
                         PyFloat_FromDouble(control->infinity));
    PyDict_SetItemString(py_options, "stop_p",
                         PyFloat_FromDouble(control->stop_p));
    PyDict_SetItemString(py_options, "stop_d",
                         PyFloat_FromDouble(control->stop_d));
    PyDict_SetItemString(py_options, "stop_c",
                         PyFloat_FromDouble(control->stop_c));
    PyDict_SetItemString(py_options, "theta_d",
                         PyFloat_FromDouble(control->theta_d));
    PyDict_SetItemString(py_options, "theta_c",
                         PyFloat_FromDouble(control->theta_c));
    PyDict_SetItemString(py_options, "beta",
                         PyFloat_FromDouble(control->beta));
    PyDict_SetItemString(py_options, "prfeas",
                         PyFloat_FromDouble(control->prfeas));
    PyDict_SetItemString(py_options, "dufeas",
                         PyFloat_FromDouble(control->dufeas));
    PyDict_SetItemString(py_options, "muzero",
                         PyFloat_FromDouble(control->muzero));
    PyDict_SetItemString(py_options, "reduce_infeas",
                         PyFloat_FromDouble(control->reduce_infeas));
    PyDict_SetItemString(py_options, "obj_unbounded",
                         PyFloat_FromDouble(control->obj_unbounded));
    PyDict_SetItemString(py_options, "pivot_tol",
                         PyFloat_FromDouble(control->pivot_tol));
    PyDict_SetItemString(py_options, "pivot_tol_for_dependencies",
                         PyFloat_FromDouble(control->pivot_tol_for_dependencies));
    PyDict_SetItemString(py_options, "zero_pivot",
                         PyFloat_FromDouble(control->zero_pivot));
    PyDict_SetItemString(py_options, "identical_bounds_tol",
                         PyFloat_FromDouble(control->identical_bounds_tol));
    PyDict_SetItemString(py_options, "inner_stop_relative",
                         PyFloat_FromDouble(control->inner_stop_relative));
    PyDict_SetItemString(py_options, "inner_stop_absolute",
                         PyFloat_FromDouble(control->inner_stop_absolute));
    PyDict_SetItemString(py_options, "initial_radius",
                         PyFloat_FromDouble(control->initial_radius));
    PyDict_SetItemString(py_options, "mu_min",
                         PyFloat_FromDouble(control->mu_min));
    PyDict_SetItemString(py_options, "inner_fraction_opt",
                         PyFloat_FromDouble(control->inner_fraction_opt));
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
    PyDict_SetItemString(py_options, "center",
                         PyBool_FromLong(control->center));
    PyDict_SetItemString(py_options, "primal",
                         PyBool_FromLong(control->primal));
    PyDict_SetItemString(py_options, "puiseux",
                         PyBool_FromLong(control->puiseux));
    PyDict_SetItemString(py_options, "feasol",
                         PyBool_FromLong(control->feasol));
    PyDict_SetItemString(py_options, "array_syntax_worse_than_do_loop",
                         PyBool_FromLong(control->array_syntax_worse_than_do_loop));
    PyDict_SetItemString(py_options, "space_critical",
                         PyBool_FromLong(control->space_critical));
    PyDict_SetItemString(py_options, "deallocate_error_fatal",
                         PyBool_FromLong(control->deallocate_error_fatal));
    PyDict_SetItemString(py_options, "generate_sif_file",
                         PyBool_FromLong(control->generate_sif_file));
    PyDict_SetItemString(py_options, "sif_file_name",
                         PyUnicode_FromString(control->sif_file_name));
    PyDict_SetItemString(py_options, "prefix",
                         PyUnicode_FromString(control->prefix));
    PyDict_SetItemString(py_options, "lsqp_options",
                         lsqp_make_options_dict(&control->lsqp_control));
    PyDict_SetItemString(py_options, "fdc_options",
                         fdc_make_options_dict(&control->fdc_control));
    PyDict_SetItemString(py_options, "sbls_options",
                         sbls_make_options_dict(&control->sbls_control));
    PyDict_SetItemString(py_options, "gltr_options",
                         gltr_make_options_dict(&control->gltr_control));
    PyDict_SetItemString(py_options, "fit_options",
                         fit_make_options_dict(&control->fit_control));

    return py_options;
}


//  *-*-*-*-*-*-*-*-*-*-   MAKE TIME    -*-*-*-*-*-*-*-*-*-*

/* Take the time struct from C and turn it into a python dictionary */
static PyObject* qpb_make_time_dict(const struct qpb_time_type *time){
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
    PyDict_SetItemString(py_time, "phase1_total",
                         PyFloat_FromDouble(time->phase1_total));
    PyDict_SetItemString(py_time, "phase1_analyse",
                         PyFloat_FromDouble(time->phase1_analyse));
    PyDict_SetItemString(py_time, "phase1_factorize",
                         PyFloat_FromDouble(time->phase1_factorize));
    PyDict_SetItemString(py_time, "phase1_solve",
                         PyFloat_FromDouble(time->phase1_solve));
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
    PyDict_SetItemString(py_time, "clock_phase1_total",
                         PyFloat_FromDouble(time->clock_phase1_total));
    PyDict_SetItemString(py_time, "clock_phase1_analyse",
                         PyFloat_FromDouble(time->clock_phase1_analyse));
    PyDict_SetItemString(py_time, "clock_phase1_factorize",
                         PyFloat_FromDouble(time->clock_phase1_factorize));
    PyDict_SetItemString(py_time, "clock_phase1_solve",
                         PyFloat_FromDouble(time->clock_phase1_solve));

    return py_time;
}

//  *-*-*-*-*-*-*-*-*-*-   MAKE INFORM    -*-*-*-*-*-*-*-*-*-*

/* Take the inform struct from C and turn it into a python dictionary */
static PyObject* qpb_make_inform_dict(const struct qpb_inform_type *inform){
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
    PyDict_SetItemString(py_inform, "nbacts",
                         PyLong_FromLong(inform->nbacts));
    PyDict_SetItemString(py_inform, "nmods",
                         PyLong_FromLong(inform->nmods));
    PyDict_SetItemString(py_inform, "obj",
                         PyFloat_FromDouble(inform->obj));
    PyDict_SetItemString(py_inform, "non_negligible_pivot",
                         PyFloat_FromDouble(inform->non_negligible_pivot));
    PyDict_SetItemString(py_inform, "feasible",
                         PyBool_FromLong(inform->feasible));

    // Set time nested dictionary
    PyDict_SetItemString(py_inform, "time",
                         qpb_make_time_dict(&inform->time));

    // Set dictionaries from subservient packages
    PyDict_SetItemString(py_inform, "lsqp_inform",
                         lsqp_make_inform_dict(&inform->lsqp_inform));
    PyDict_SetItemString(py_inform, "fdc_inform",
                         fdc_make_inform_dict(&inform->fdc_inform));
    PyDict_SetItemString(py_inform, "sbls_inform",
                        sbls_make_inform_dict(&inform->sbls_inform));
    PyDict_SetItemString(py_inform, "gltr_inform",
                         gltr_make_inform_dict(&inform->gltr_inform));
    PyDict_SetItemString(py_inform, "fit_inform",
                         fit_make_inform_dict(&inform->fit_inform));

    return py_inform;
}

//  *-*-*-*-*-*-*-*-*-*-   QPB_INITIALIZE    -*-*-*-*-*-*-*-*-*-*

static PyObject* py_qpb_initialize(PyObject *self){

    // Call qpb_initialize
    qpb_initialize(&data, &control, &status);

    // Record that QPB has been initialised
    init_called = true;

    // Return options Python dictionary
    PyObject *py_options = qpb_make_options_dict(&control);
    return Py_BuildValue("O", py_options);
}

//  *-*-*-*-*-*-*-*-*-*-*-*-   QPB_LOAD    -*-*-*-*-*-*-*-*-*-*-*-*

static PyObject* py_qpb_load(PyObject *self, PyObject *args, PyObject *keywds){
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
    qpb_reset_control(&control, &data, &status);

    // Update QPB control options
    if(!qpb_update_control(&control, py_options))
        return NULL;

    // Call qpb_import
    qpb_import(&control, &data, &status, n, m,
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

//  *-*-*-*-*-*-*-*-*-*-   QPB_SOLVE_QP   -*-*-*-*-*-*-*-*

static PyObject* py_qpb_solve_qp(PyObject *self, PyObject *args, PyObject *keywds){
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
    if(!PyArg_ParseTupleAndKeywords(args, keywds, "iidOiOiOOOOOOOO", kwlist, &n, &m, &f, &py_g,
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

    // Call qpb_solve_direct
    status = 1; // set status to 1 on entry
    qpb_solve_qp(&data, &status, n, m, H_ne, H_val, g, f, A_ne, A_val,
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

//  *-*-*-*-*-*-*-*-*-*-   QPB_INFORMATION   -*-*-*-*-*-*-*-*

static PyObject* py_qpb_information(PyObject *self){

    // Check that package has been initialised
    if(!check_init(init_called))
        return NULL;

    // Call qpb_information
    qpb_information(&data, &inform, &status);

    // Return status and inform Python dictionary
    PyObject *py_inform = qpb_make_inform_dict(&inform);
    return Py_BuildValue("O", py_inform);
}

//  *-*-*-*-*-*-*-*-*-*-   QPB_TERMINATE   -*-*-*-*-*-*-*-*-*-*

static PyObject* py_qpb_terminate(PyObject *self){

    // Check that package has been initialised
    if(!check_init(init_called))
        return NULL;

    // Call qpb_terminate
    qpb_terminate(&data, &control, &inform);

    // Return None boilerplate
    Py_INCREF(Py_None);
    return Py_None;
}

//  *-*-*-*-*-*-*-*-*-*-   INITIALIZE QPB PYTHON MODULE    -*-*-*-*-*-*-*-*-*-*

/* qpb python module method table */
static PyMethodDef qpb_module_methods[] = {
    {"initialize", (PyCFunction) py_qpb_initialize, METH_NOARGS, NULL},
    {"load", (PyCFunction) py_qpb_load, METH_VARARGS | METH_KEYWORDS, NULL},
    {"solve_qp", (PyCFunction) py_qpb_solve_qp, METH_VARARGS | METH_KEYWORDS, NULL},
    {"information", (PyCFunction) py_qpb_information, METH_NOARGS, NULL},
    {"terminate", (PyCFunction) py_qpb_terminate, METH_NOARGS, NULL},
    {NULL, NULL, 0, NULL}  /* Sentinel */
};

/* qpb python module documentation */

PyDoc_STRVAR(qpb_module_doc,

"The qpa package uses a primal-dual interior-point method to solve the \n"
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
"See $GALAHAD/html/Python/qpa.html for argument lists, call order\n"
"and other details.\n"
"\n"
);

/* qpb python module definition */
static struct PyModuleDef module = {
   PyModuleDef_HEAD_INIT,
   "qpb",               /* name of module */
   qpb_module_doc,      /* module documentation, may be NULL */
   -1,                  /* size of per-interpreter state of the module,or -1
                           if the module keeps state in global variables */
   qpb_module_methods   /* module methods */
};

/* Python module initialization */
PyMODINIT_FUNC PyInit_qpb(void) { // must be same as module name above
    import_array();  // for NumPy arrays
    return PyModule_Create(&module);
}
