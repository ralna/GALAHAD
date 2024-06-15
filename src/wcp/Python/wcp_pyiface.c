//* \file wcp_pyiface.c */

/*
 * THIS VERSION: GALAHAD 4.1 - 2023-05-20 AT 10:10 GMT.
 *
 *-*-*-*-*-*-*-*-*-  GALAHAD_WCP PYTHON INTERFACE  *-*-*-*-*-*-*-*-*-*-
 *
 *  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
 *  Principal author: Jaroslav Fowkes & Nick Gould
 *
 *  History -
 *   originally released GALAHAD Version 4.1. April 120th 2023
 *
 *  For full documentation, see
 *   http://galahad.rl.ac.uk/galahad-www/specs.html
 */

#include "galahad_python.h"
#include "galahad_wcp.h"

/* Nested FDC and SBLS control and inform prototypes */
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
static struct wcp_control_type control;  // control struct
static struct wcp_inform_type inform;    // inform struct
static bool init_called = false;         // record if initialise was called
static int status = 0;                   // exit status

//  *-*-*-*-*-*-*-*-*-*-   UPDATE CONTROL    -*-*-*-*-*-*-*-*-*-*

/* Update the control options: use C defaults but update any passed via Python*/
static bool wcp_update_control(struct wcp_control_type *control,
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
        if(strcmp(key_name, "initial_point") == 0){
            if(!parse_int_option(value, "initial_point",
                                  &control->initial_point))
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
        if(strcmp(key_name, "perturbation_strategy") == 0){
            if(!parse_int_option(value, "perturbation_strategy",
                                  &control->perturbation_strategy))
                return false;
            continue;
        }
        if(strcmp(key_name, "restore_problem") == 0){
            if(!parse_int_option(value, "restore_problem",
                                  &control->restore_problem))
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
        if(strcmp(key_name, "mu_target") == 0){
            if(!parse_double_option(value, "mu_target",
                                  &control->mu_target))
                return false;
            continue;
        }
        if(strcmp(key_name, "mu_accept_fraction") == 0){
            if(!parse_double_option(value, "mu_accept_fraction",
                                  &control->mu_accept_fraction))
                return false;
            continue;
        }
        if(strcmp(key_name, "mu_increase_factor") == 0){
            if(!parse_double_option(value, "mu_increase_factor",
                                  &control->mu_increase_factor))
                return false;
            continue;
        }
        if(strcmp(key_name, "required_infeas_reduction") == 0){
            if(!parse_double_option(value, "required_infeas_reduction",
                                  &control->required_infeas_reduction))
                return false;
            continue;
        }
        if(strcmp(key_name, "implicit_tol") == 0){
            if(!parse_double_option(value, "implicit_tol",
                                  &control->implicit_tol))
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
        if(strcmp(key_name, "perturb_start") == 0){
            if(!parse_double_option(value, "perturb_start",
                                  &control->perturb_start))
                return false;
            continue;
        }
        if(strcmp(key_name, "alpha_scale") == 0){
            if(!parse_double_option(value, "alpha_scale",
                                  &control->alpha_scale))
                return false;
            continue;
        }
        if(strcmp(key_name, "identical_bounds_tol") == 0){
            if(!parse_double_option(value, "identical_bounds_tol",
                                  &control->identical_bounds_tol))
                return false;
            continue;
        }
        if(strcmp(key_name, "reduce_perturb_factor") == 0){
            if(!parse_double_option(value, "reduce_perturb_factor",
                                  &control->reduce_perturb_factor))
                return false;
            continue;
        }
        if(strcmp(key_name, "reduce_perturb_multiplier") == 0){
            if(!parse_double_option(value, "reduce_perturb_multiplier",
                                  &control->reduce_perturb_multiplier))
                return false;
            continue;
        }
        if(strcmp(key_name, "insufficiently_feasible") == 0){
            if(!parse_double_option(value, "insufficiently_feasible",
                                  &control->insufficiently_feasible))
                return false;
            continue;
        }
        if(strcmp(key_name, "perturbation_small") == 0){
            if(!parse_double_option(value, "perturbation_small",
                                  &control->perturbation_small))
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
        if(strcmp(key_name, "balance_initial_complementarity") == 0){
            if(!parse_bool_option(value, "balance_initial_complementarity",
                                  &control->balance_initial_complementarity))
                return false;
            continue;
        }
        if(strcmp(key_name, "use_corrector") == 0){
            if(!parse_bool_option(value, "use_corrector",
                                  &control->use_corrector))
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
        if(strcmp(key_name, "record_x_status") == 0){
            if(!parse_bool_option(value, "record_x_status",
                                  &control->record_x_status))
                return false;
            continue;
        }
        if(strcmp(key_name, "record_c_status") == 0){
            if(!parse_bool_option(value, "record_c_status",
                                  &control->record_c_status))
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
// NB not static as it is used for nested inform within QP Python interface
PyObject* wcp_make_options_dict(const struct wcp_control_type *control){
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
    PyDict_SetItemString(py_options, "initial_point",
                         PyLong_FromLong(control->initial_point));
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
    PyDict_SetItemString(py_options, "perturbation_strategy",
                         PyLong_FromLong(control->perturbation_strategy));
    PyDict_SetItemString(py_options, "restore_problem",
                         PyLong_FromLong(control->restore_problem));
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
    PyDict_SetItemString(py_options, "mu_target",
                         PyFloat_FromDouble(control->mu_target));
    PyDict_SetItemString(py_options, "mu_accept_fraction",
                         PyFloat_FromDouble(control->mu_accept_fraction));
    PyDict_SetItemString(py_options, "mu_increase_factor",
                         PyFloat_FromDouble(control->mu_increase_factor));
    PyDict_SetItemString(py_options, "required_infeas_reduction",
                         PyFloat_FromDouble(control->required_infeas_reduction));
    PyDict_SetItemString(py_options, "implicit_tol",
                         PyFloat_FromDouble(control->implicit_tol));
    PyDict_SetItemString(py_options, "pivot_tol",
                         PyFloat_FromDouble(control->pivot_tol));
    PyDict_SetItemString(py_options, "pivot_tol_for_dependencies",
                         PyFloat_FromDouble(control->pivot_tol_for_dependencies));
    PyDict_SetItemString(py_options, "zero_pivot",
                         PyFloat_FromDouble(control->zero_pivot));
    PyDict_SetItemString(py_options, "perturb_start",
                         PyFloat_FromDouble(control->perturb_start));
    PyDict_SetItemString(py_options, "alpha_scale",
                         PyFloat_FromDouble(control->alpha_scale));
    PyDict_SetItemString(py_options, "identical_bounds_tol",
                         PyFloat_FromDouble(control->identical_bounds_tol));
    PyDict_SetItemString(py_options, "reduce_perturb_factor",
                         PyFloat_FromDouble(control->reduce_perturb_factor));
    PyDict_SetItemString(py_options, "reduce_perturb_multiplier",
                         PyFloat_FromDouble(control->reduce_perturb_multiplier));
    PyDict_SetItemString(py_options, "insufficiently_feasible",
                         PyFloat_FromDouble(control->insufficiently_feasible));
    PyDict_SetItemString(py_options, "perturbation_small",
                         PyFloat_FromDouble(control->perturbation_small));
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
    PyDict_SetItemString(py_options, "balance_initial_complementarity",
                         PyBool_FromLong(control->balance_initial_complementarity));
    PyDict_SetItemString(py_options, "use_corrector",
                         PyBool_FromLong(control->use_corrector));
    PyDict_SetItemString(py_options, "space_critical",
                         PyBool_FromLong(control->space_critical));
    PyDict_SetItemString(py_options, "deallocate_error_fatal",
                         PyBool_FromLong(control->deallocate_error_fatal));
    PyDict_SetItemString(py_options, "record_x_status",
                         PyBool_FromLong(control->record_x_status));
    PyDict_SetItemString(py_options, "record_c_status",
                         PyBool_FromLong(control->record_c_status));
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
static PyObject* wcp_make_time_dict(const struct wcp_time_type *time){
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
static PyObject* wcp_make_inform_dict(const struct wcp_inform_type *inform){
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
    PyDict_SetItemString(py_inform, "c_implicit",
                         PyLong_FromLong(inform->c_implicit));
    PyDict_SetItemString(py_inform, "x_implicit",
                         PyLong_FromLong(inform->x_implicit));
    PyDict_SetItemString(py_inform, "y_implicit",
                         PyLong_FromLong(inform->y_implicit));
    PyDict_SetItemString(py_inform, "z_implicit",
                         PyLong_FromLong(inform->z_implicit));
    PyDict_SetItemString(py_inform, "obj",
                         PyFloat_FromDouble(inform->obj));
    PyDict_SetItemString(py_inform, "mu_final_target_max",
                         PyFloat_FromDouble(inform->mu_final_target_max));
    PyDict_SetItemString(py_inform, "non_negligible_pivot",
                         PyFloat_FromDouble(inform->non_negligible_pivot));
    PyDict_SetItemString(py_inform, "feasible",
                         PyBool_FromLong(inform->feasible));

    // Set time nested dictionary
    PyDict_SetItemString(py_inform, "time",
                         wcp_make_time_dict(&inform->time));

    // Set dictionaries from subservient packages
    PyDict_SetItemString(py_inform, "fdc_inform",
                         fdc_make_inform_dict(&inform->fdc_inform));
    PyDict_SetItemString(py_inform, "sbls_inform",
                         sbls_make_inform_dict(&inform->sbls_inform));

    return py_inform;
}

//  *-*-*-*-*-*-*-*-*-*-   WCP_INITIALIZE    -*-*-*-*-*-*-*-*-*-*

static PyObject* py_wcp_initialize(PyObject *self){

    // Call wcp_initialize
    wcp_initialize(&data, &control, &status);

    // Record that WCP has been initialised
    init_called = true;

    // Return options Python dictionary
    PyObject *py_options = wcp_make_options_dict(&control);
    return Py_BuildValue("O", py_options);
}

//  *-*-*-*-*-*-*-*-*-*-*-*-   WCP_LOAD    -*-*-*-*-*-*-*-*-*-*-*-*

static PyObject* py_wcp_load(PyObject *self, PyObject *args, PyObject *keywds){
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
    wcp_reset_control(&control, &data, &status);

    // Update WCP control options
    if(!wcp_update_control(&control, py_options))
        return NULL;

    // Call wcp_import
    wcp_import(&control, &data, &status, n, m,
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

//  *-*-*-*-*-*-*-*-*-*-   WCP_FIND_WCP   -*-*-*-*-*-*-*-*

static PyObject* py_wcp_find_wcp(PyObject *self, PyObject *args, PyObject *keywds){
    PyArrayObject *py_A_val;
    PyArrayObject *py_c_l, *py_c_u, *py_x_l, *py_x_u;
    PyArrayObject *py_x, *py_y_l, *py_y_u, *py_z_l, *py_z_u, *py_g;
    double *A_val, *c_l, *c_u, *x_l, *x_u, *x, *y_l, *y_u, *z_l, *z_u;
    double *g = NULL;
    int n, m, A_ne;

    // Check that package has been initialised
    if(!check_init(init_called))
        return NULL;

    // Parse positional arguments
    static char *kwlist[] = {"n", "m", "A_ne", "A_val", "c_l", "c_u", "x_l", "x_u",
                             "x", "y_l", "y_u", "z_l", "z_u", "g", NULL};
    if(!PyArg_ParseTupleAndKeywords(args, keywds, "iiiOOOOOOOOOO|O", kwlist, &n, &m,
                                    &A_ne, &py_A_val, &py_c_l, &py_c_u, &py_x_l, &py_x_u,
                                    &py_x, &py_y_l, &py_y_u, &py_z_l, &py_z_u, &py_g))
        return NULL;

    // Check that array inputs are of correct type, size, and shape
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
    if(!check_array_double("y_l", py_y_l, m))
        return NULL;
    if(!check_array_double("y_u", py_y_u, m))
        return NULL;
    if(!check_array_double("z_l", py_z_l, n))
        return NULL;
    if(!check_array_double("z_u", py_z_u, n))
        return NULL;
    if((PyObject *) py_g == Py_None){
      g = malloc(n * sizeof(double));
      for(int i = 0; i < n; i++) g[i] = 0.0;
    } else {
      g = (double *) PyArray_DATA(py_g);
    }

    // Get array data pointer
    A_val = (double *) PyArray_DATA(py_A_val);
    c_l = (double *) PyArray_DATA(py_c_l);
    c_u = (double *) PyArray_DATA(py_c_u);
    x_l = (double *) PyArray_DATA(py_x_l);
    x_u = (double *) PyArray_DATA(py_x_u);
    x = (double *) PyArray_DATA(py_x);
    y_l = (double *) PyArray_DATA(py_y_l);
    y_u = (double *) PyArray_DATA(py_y_u);
    z_l = (double *) PyArray_DATA(py_z_l);
    z_u = (double *) PyArray_DATA(py_z_u);

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

    // Call wcp_solve_direct
    status = 1; // set status to 1 on entry
    wcp_find_wcp(&data, &status, n, m, g, A_ne, A_val,
                 c_l, c_u, x_l, x_u, x, c, y_l, y_u, z_l, z_u, x_stat, c_stat);
    // for( int i = 0; i < n; i++) printf("x %f\n", x[i]);
    // for( int i = 0; i < m; i++) printf("c %f\n", c[i]);
    // for( int i = 0; i < n; i++) printf("x_stat %i\n", x_stat[i]);
    // for( int i = 0; i < m; i++) printf("c_stat %i\n", c_stat[i]);

    // Free allocated memory
    if(py_g == NULL) free(g);

    // Propagate any errors with the callback function
    if(PyErr_Occurred())
        return NULL;

    // Raise any status errors
    if(!check_error_codes(status))
        return NULL;

    // Return x, c, y, z, x_stat and c_stat
    PyObject *find_wcp_return;

    // find_wcp_return = Py_BuildValue("O", py_x);
    find_wcp_return = Py_BuildValue("OOOOOOOO", py_x, py_c, py_y_l, py_y_u,
                                                py_z_l, py_z_u,
                                                py_x_stat, py_c_stat);
    Py_INCREF(find_wcp_return);
    return find_wcp_return;
}

//  *-*-*-*-*-*-*-*-*-*-   WCP_INFORMATION   -*-*-*-*-*-*-*-*

static PyObject* py_wcp_information(PyObject *self){

    // Check that package has been initialised
    if(!check_init(init_called))
        return NULL;

    // Call wcp_information
    wcp_information(&data, &inform, &status);

    // Return status and inform Python dictionary
    PyObject *py_inform = wcp_make_inform_dict(&inform);
    return Py_BuildValue("O", py_inform);
}

//  *-*-*-*-*-*-*-*-*-*-   WCP_TERMINATE   -*-*-*-*-*-*-*-*-*-*

static PyObject* py_wcp_terminate(PyObject *self){

    // Check that package has been initialised
    if(!check_init(init_called))
        return NULL;

    // Call wcp_terminate
    wcp_terminate(&data, &control, &inform);

    // Return None boilerplate
    Py_INCREF(Py_None);
    return Py_None;
}

//  *-*-*-*-*-*-*-*-*-*-   INITIALIZE WCP PYTHON MODULE    -*-*-*-*-*-*-*-*-*-*

/* wcp python module method table */
static PyMethodDef wcp_module_methods[] = {
    {"initialize", (PyCFunction) py_wcp_initialize, METH_NOARGS, NULL},
    {"load", (PyCFunction) py_wcp_load, METH_VARARGS | METH_KEYWORDS, NULL},
    {"find_wcp", (PyCFunction) py_wcp_find_wcp, METH_VARARGS | METH_KEYWORDS, NULL},
    {"information", (PyCFunction) py_wcp_information, METH_NOARGS, NULL},
    {"terminate", (PyCFunction) py_wcp_terminate, METH_NOARGS, NULL},
    {NULL, NULL, 0, NULL}  /* Sentinel */
};

/* wcp python module documentation */

PyDoc_STRVAR(wcp_module_doc,

"The wcp package uses a primal-dual interior-point method to find a\n"
"well-centered point within a polyhedral set.\n"
"The aim is to find a point that lies interior to the boundary of the\n"
"polyhedron definied by the general linear constraints and simple bounds\n"
"c_l  <=  A x  <= c_u and x_l  <=  x  <= x_u\n"
"where A is a given m by n matrix, and any of the components\n"
"of the vectors c_l, c_u, x_l or x_u may be infinite.\n"
"The method offers the choice of direct and iterative solution of the key\n"
"regularization subproblems, and is most suitable for problems\n"
"involving a large number of unknowns x, since full advantage is taken of\n"
"any zero coefficients in the matrix A.\n"
"The package identifies infeasible problems, and problems for which there\n"
"is no strict interior.\n"
"\n"
"See $GALAHAD/html/Python/wcp.html for argument lists, call order\n"
"and other details.\n"
"\n"
);

/* wcp python module definition */
static struct PyModuleDef module = {
   PyModuleDef_HEAD_INIT,
   "wcp",               /* name of module */
   wcp_module_doc,      /* module documentation, may be NULL */
   -1,                  /* size of per-interpreter state of the module,or -1
                           if the module keeps state in global variables */
   wcp_module_methods   /* module methods */
};

/* Python module initialization */
PyMODINIT_FUNC PyInit_wcp(void) { // must be same as module name above
    import_array();  // for NumPy arrays
    return PyModule_Create(&module);
}
