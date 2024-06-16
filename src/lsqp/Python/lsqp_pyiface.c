//* \file lsqp_pyiface.c */

/*
 * THIS VERSION: GALAHAD 5.0 - 2024-06-15 AT 11:50 GMT.
 *
 *-*-*-*-*-*-*-*-*-  GALAHAD_LSQP PYTHON INTERFACE  *-*-*-*-*-*-*-*-*-*-
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
#include "galahad_lsqp.h"

/* Nested FDC & SBLS control and inform prototypes */
bool fdc_update_control(struct fdc_control_type *control,
                        PyObject *py_options);
PyObject* fdc_make_options_dict(const struct fdc_control_type *control);
PyObject* fdc_make_inform_dict(const struct fdc_inform_type *inform);
bool sbls_update_control(struct sbls_control_type *control,
                         PyObject *py_options);
PyObject* sbls_make_options_dict(const struct sbls_control_type *control);
PyObject* sbls_make_inform_dict(const struct sbls_inform_type *inform);

/* Module global variables */
static void *data;                       // private internal data
static struct lsqp_control_type control;  // control struct
static struct lsqp_inform_type inform;    // inform struct
static bool init_called = false;         // record if initialise was called
static int status = 0;                   // exit status

//  *-*-*-*-*-*-*-*-*-*-   UPDATE CONTROL    -*-*-*-*-*-*-*-*-*-*

/* Update the control options: use C defaults but update any passed via Python*/
// NB not static as it is used for nested control within QPB Python interface
bool lsqp_update_control(struct lsqp_control_type *control,
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
        if(strcmp(key_name, "itref_max") == 0){
            if(!parse_int_option(value, "itref_max",
                                  &control->itref_max))
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
        if(strcmp(key_name, "potential_unbounded") == 0){
            if(!parse_double_option(value, "potential_unbounded",
                                  &control->potential_unbounded))
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
        if(strcmp(key_name, "mu_min") == 0){
            if(!parse_double_option(value, "mu_min",
                                  &control->mu_min))
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
        if(strcmp(key_name, "use_corrector") == 0){
            if(!parse_bool_option(value, "use_corrector",
                                  &control->use_corrector))
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

        // Otherwise unrecognised option
        PyErr_Format(PyExc_ValueError,
          "unrecognised option options['%s']\n", key_name);
        return false;
    }

    return true; // success
}

//  *-*-*-*-*-*-*-*-*-*-   MAKE OPTIONS    -*-*-*-*-*-*-*-*-*-*

/* Take the control struct from C and turn it into a python options dict */
// NB not static as it is used for nested inform within QPB Python interface
PyObject* lsqp_make_options_dict(const struct lsqp_control_type *control){
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
    PyDict_SetItemString(py_options, "factor",
                         PyLong_FromLong(control->factor));
    PyDict_SetItemString(py_options, "max_col",
                         PyLong_FromLong(control->max_col));
    PyDict_SetItemString(py_options, "indmin",
                         PyLong_FromLong(control->indmin));
    PyDict_SetItemString(py_options, "valmin",
                         PyLong_FromLong(control->valmin));
    PyDict_SetItemString(py_options, "itref_max",
                         PyLong_FromLong(control->itref_max));
    PyDict_SetItemString(py_options, "infeas_max",
                         PyLong_FromLong(control->infeas_max));
    PyDict_SetItemString(py_options, "muzero_fixed",
                         PyLong_FromLong(control->muzero_fixed));
    PyDict_SetItemString(py_options, "restore_problem",
                         PyLong_FromLong(control->restore_problem));
    PyDict_SetItemString(py_options, "indicator_type",
                         PyLong_FromLong(control->indicator_type));
    PyDict_SetItemString(py_options, "extrapolate",
                         PyLong_FromLong(control->extrapolate));
    PyDict_SetItemString(py_options, "path_history",
                         PyLong_FromLong(control->path_history));
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
    PyDict_SetItemString(py_options, "prfeas",
                         PyFloat_FromDouble(control->prfeas));
    PyDict_SetItemString(py_options, "dufeas",
                         PyFloat_FromDouble(control->dufeas));
    PyDict_SetItemString(py_options, "muzero",
                         PyFloat_FromDouble(control->muzero));
    PyDict_SetItemString(py_options, "reduce_infeas",
                         PyFloat_FromDouble(control->reduce_infeas));
    PyDict_SetItemString(py_options, "potential_unbounded",
                         PyFloat_FromDouble(control->potential_unbounded));
    PyDict_SetItemString(py_options, "pivot_tol",
                         PyFloat_FromDouble(control->pivot_tol));
    PyDict_SetItemString(py_options, "pivot_tol_for_dependencies",
                         PyFloat_FromDouble(control->pivot_tol_for_dependencies));
    PyDict_SetItemString(py_options, "zero_pivot",
                         PyFloat_FromDouble(control->zero_pivot));
    PyDict_SetItemString(py_options, "identical_bounds_tol",
                         PyFloat_FromDouble(control->identical_bounds_tol));
    PyDict_SetItemString(py_options, "mu_min",
                         PyFloat_FromDouble(control->mu_min));
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
    PyDict_SetItemString(py_options, "just_feasible",
                         PyBool_FromLong(control->just_feasible));
    PyDict_SetItemString(py_options, "getdua",
                         PyBool_FromLong(control->getdua));
    PyDict_SetItemString(py_options, "puiseux",
                         PyBool_FromLong(control->puiseux));
    PyDict_SetItemString(py_options, "feasol",
                         PyBool_FromLong(control->feasol));
    PyDict_SetItemString(py_options, "balance_initial_complentarity",
                         PyBool_FromLong(control->balance_initial_complentarity));
    PyDict_SetItemString(py_options, "use_corrector",
                         PyBool_FromLong(control->use_corrector));
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
    PyDict_SetItemString(py_options, "fdc_options",
                         fdc_make_options_dict(&control->fdc_control));
    PyDict_SetItemString(py_options, "sbls_options",
                         sbls_make_options_dict(&control->sbls_control));

    return py_options;
}


//  *-*-*-*-*-*-*-*-*-*-   MAKE TIME    -*-*-*-*-*-*-*-*-*-*

/* Take the time struct from C and turn it into a python dictionary */
static PyObject* lsqp_make_time_dict(const struct lsqp_time_type *time){
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
// NB not static as it is used for nested control within QPB Python interface
PyObject* lsqp_make_inform_dict(const struct lsqp_inform_type *inform){
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
    PyDict_SetItemString(py_inform, "obj",
                         PyFloat_FromDouble(inform->obj));
    PyDict_SetItemString(py_inform, "potential",
                         PyFloat_FromDouble(inform->potential));
    PyDict_SetItemString(py_inform, "non_negligible_pivot",
                         PyFloat_FromDouble(inform->non_negligible_pivot));
    PyDict_SetItemString(py_inform, "feasible",
                         PyBool_FromLong(inform->feasible));

    // Set time nested dictionary
    PyDict_SetItemString(py_inform, "time",
                         lsqp_make_time_dict(&inform->time));

    // Set dictionaries from subservient packages
    PyDict_SetItemString(py_inform, "fdc_inform",
                         fdc_make_inform_dict(&inform->fdc_inform));
    PyDict_SetItemString(py_inform, "sbls_inform",
                        sbls_make_inform_dict(&inform->sbls_inform));

    return py_inform;
}

//  *-*-*-*-*-*-*-*-*-*-   LSQP_INITIALIZE    -*-*-*-*-*-*-*-*-*-*

static PyObject* py_lsqp_initialize(PyObject *self){

    // Call lsqp_initialize
    lsqp_initialize(&data, &control, &status);

    // Record that LSQP has been initialised
    init_called = true;

    // Return options Python dictionary
    PyObject *py_options = lsqp_make_options_dict(&control);
    return Py_BuildValue("O", py_options);
}

//  *-*-*-*-*-*-*-*-*-*-*-*-   LSQP_LOAD    -*-*-*-*-*-*-*-*-*-*-*-*

static PyObject* py_lsqp_load(PyObject *self, PyObject *args, PyObject *keywds){
    PyArrayObject *py_A_row, *py_A_col, *py_A_ptr;
    PyObject *py_options = NULL;
    int *A_row = NULL, *A_col = NULL, *A_ptr = NULL;
    const char *A_type;
    int n, m, A_ne;

    // Check that package has been initialised
    if(!check_init(init_called))
        return NULL;

    // Parse positional and keyword arguments
    static char *kwlist[] = {"n","m","A_type","A_ne","A_row","A_col","A_ptr",
                             "options",NULL};

    if(!PyArg_ParseTupleAndKeywords(args, keywds, "iisiOOO|O",
                                    kwlist, &n, &m,
                                    &A_type, &A_ne, &py_A_row,
                                    &py_A_col, &py_A_ptr,
                                    &py_options))
        return NULL;

    // Check that array inputs are of correct type, size, and shape

    if(!(
        check_array_int("A_row", py_A_row, A_ne) &&
        check_array_int("A_col", py_A_col, A_ne) &&
        check_array_int("A_ptr", py_A_ptr, m+1)
        ))
        return NULL;

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
    lsqp_reset_control(&control, &data, &status);

    // Update LSQP control options
    if(!lsqp_update_control(&control, py_options))
        return NULL;

    // Call lsqp_import
    lsqp_import(&control, &data, &status, n, m,
               A_type, A_ne, A_row, A_col, A_ptr);

    // Free allocated memory
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

//  *-*-*-*-*-*-*-*-*-*-   LSQP_SOLVE_QP   -*-*-*-*-*-*-*-*

static PyObject* py_lsqp_solve_qp(PyObject *self, PyObject *args, PyObject *keywds){
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
    if(!PyArg_ParseTupleAndKeywords(args, keywds, "iidOOOiOOOOOOOO", kwlist, &n, &m, &f, &py_g,
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

    // Call lsqp_solve_direct
    status = 1; // set status to 1 on entry
    lsqp_solve_qp(&data, &status, n, m, w, x0, g, f, A_ne, A_val,
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
    solve_qp_return = Py_BuildValue("OOOOOO", py_x, py_c, py_y, py_z,
                                                 py_x_stat, py_c_stat);
    Py_INCREF(solve_qp_return);
    return solve_qp_return;
}

//  *-*-*-*-*-*-*-*-*-*-   LSQP_INFORMATION   -*-*-*-*-*-*-*-*

static PyObject* py_lsqp_information(PyObject *self){

    // Check that package has been initialised
    if(!check_init(init_called))
        return NULL;

    // Call lsqp_information
    lsqp_information(&data, &inform, &status);

    // Return status and inform Python dictionary
    PyObject *py_inform = lsqp_make_inform_dict(&inform);
    return Py_BuildValue("O", py_inform);
}

//  *-*-*-*-*-*-*-*-*-*-   LSQP_TERMINATE   -*-*-*-*-*-*-*-*-*-*

static PyObject* py_lsqp_terminate(PyObject *self){

    // Check that package has been initialised
    if(!check_init(init_called))
        return NULL;

    // Call lsqp_terminate
    lsqp_terminate(&data, &control, &inform);

    // Return None boilerplate
    Py_INCREF(Py_None);
    return Py_None;
}

//  *-*-*-*-*-*-*-*-*-*-   INITIALIZE LSQP PYTHON MODULE    -*-*-*-*-*-*-*-*-*-*

/* lsqp python module method table */
static PyMethodDef lsqp_module_methods[] = {
    {"initialize", (PyCFunction) py_lsqp_initialize, METH_NOARGS, NULL},
    {"load", (PyCFunction) py_lsqp_load, METH_VARARGS | METH_KEYWORDS, NULL},
    {"solve_qp", (PyCFunction) py_lsqp_solve_qp, METH_VARARGS | METH_KEYWORDS, NULL},
    {"information", (PyCFunction) py_lsqp_information, METH_NOARGS, NULL},
    {"terminate", (PyCFunction) py_lsqp_terminate, METH_NOARGS, NULL},
    {NULL, NULL, 0, NULL}  /* Sentinel */
};

/* lsqp python module documentation */

PyDoc_STRVAR(lsqp_module_doc,

"The lsqp package uses an interior-point trust-region method to solve a \n"
"given linear or seperable convex quadratic programming problem.\n"
"The aim is to minimize the separable quadratic objective\n"
"s(x) = 1/2 sum_{j=1}^n w_j^2 ( x_j - x_j^0 )^2,\n"
"subject to the general linear constraints\n"
"c_i^l  <=  a_i^Tx  <= c_i^u, i = 1, ... , m\n"
"and the simple bound constraints\n"
"x_j^l  <=  x_j  <= x_j^u, j = 1, ... , n,\n"
"where the ectors g, w, x^0, a_i, c^l, c^u, x^l,\n"
"x^u and the scalar f are given.\n"
"Any of the constraint bounds c_i^l, c_i^u,\n"
"x_j^l and x_j^u may be infinite.\n"
"The method offers the choice of direct and iterative solution of the key\n"
"regularization subproblems, and is most suitable for problems\n"
"involving a large number of unknowns x.\n"
"\n"
"See $GALAHAD/html/Python/lsqp.html for argument lists, call order\n"
"and other details.\n"
"\n"
);

/* lsqp python module definition */
static struct PyModuleDef module = {
   PyModuleDef_HEAD_INIT,
   "lsqp",               /* name of module */
   lsqp_module_doc,      /* module documentation, may be NULL */
   -1,                  /* size of per-interpreter state of the module,or -1
                           if the module keeps state in global variables */
   lsqp_module_methods   /* module methods */
};

/* Python module initialization */
PyMODINIT_FUNC PyInit_lsqp(void) { // must be same as module name above
    import_array();  // for NumPy arrays
    return PyModule_Create(&module);
}
