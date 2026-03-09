//* \file snls_pyiface.c */

/*
 * THIS VERSION: GALAHAD 5.5 - 2026-03-07 AT 15:30 GMT.
 *
 *-*-*-*-*-*-*-*-*-  GALAHAD_SNLS PYTHON INTERFACE  *-*-*-*-*-*-*-*-*-*-
 *
 *  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
 *  Principal author: Jaroslav Fowkes & Nick Gould
 *
 *  History -
 *   originally released GALAHAD Version 5.1. July 14th 2024
 *
 *  For full documentation, see
 *   http://galahad.rl.ac.uk/galahad-www/specs.html
 */

#include "galahad_python.h"
#include "galahad_snls.h"

/* Nested SLLS and SLLSB control and inform prototypes */
bool slls_update_control(struct slls_control_type *control,
                         PyObject *py_options);
PyObject* slls_make_options_dict(const struct slls_control_type *control);
PyObject* slls_make_inform_dict(const struct slls_inform_type *inform);
bool sllsb_update_control(struct sllsb_control_type *control,
                         PyObject *py_options);
PyObject* sllsb_make_options_dict(const struct sllsb_control_type *control);
PyObject* sllsb_make_inform_dict(const struct sllsb_inform_type *inform);

/* Module global variables */
static void *data;                       // private internal data
static struct snls_control_type control;  // control struct
static struct snls_inform_type inform;    // inform struct
static bool init_called = false;         // record if initialise was called
static int status = 0;                   // exit status

//  *-*-*-*-*-*-*-*-*-*-   CALLBACK FUNCTIONS    -*-*-*-*-*-*-*-*-*-*

/* Python eval_* function pointers */
static PyObject *py_eval_r = NULL;
static PyObject *py_eval_jr = NULL;
static PyObject *py_eval_h = NULL;
static PyObject *snls_solve_return = NULL;
//static PyObject *py_c = NULL;
//static PyObject *py_g = NULL;

/* C eval_* function wrappers */
static int eval_r(int n, int m_r, const double x[], double r[],
                  const void *userdata){

    // Wrap input array as NumPy array
    npy_intp xdim[] = {n};
    PyObject *py_x = PyArray_SimpleNewFromData(1, xdim, NPY_DOUBLE, (void *) x);

    // Build Python argument list
    PyObject *arglist = Py_BuildValue("(O)", py_x);

    // Call Python eval_r
    PyObject *result = PyObject_CallObject(py_eval_r, arglist);
    Py_DECREF(py_x);    // Free py_x memory
    Py_DECREF(arglist); // Free arglist memory

    // Check that eval was successful
    if(!result)
        return -1;

    // Get return value data pointer and copy data intoc
    const double *rval = (double *) PyArray_DATA((PyArrayObject*) result);
    for(int i=0; i<m_r; i++) {
        r[i] = rval[i];
    }

    // Free result memory
    Py_DECREF(result);

    return 0;
}

static int eval_jr(int n, int m_r, int jrne, const double x[], double jrval[],
                   const void *userdata){

    // Wrap input array as NumPy array
    npy_intp xdim[] = {n};
    PyArrayObject *py_x = (PyArrayObject*)
       PyArray_SimpleNewFromData(1, xdim, NPY_DOUBLE, (void *) x);

    // Build Python argument list
    PyObject *arglist = Py_BuildValue("(O)", py_x);

    // Call Python eval_jr
    PyObject *result = PyObject_CallObject(py_eval_jr, arglist);
    Py_DECREF(py_x);    // Free py_x memory
    Py_DECREF(arglist); // Free arglist memory

    // Check that eval was successful
    if(!result)
        return -1;

    // Check return value is of correct type, size, and shape
    if(!check_array_double("eval_jr return value",
                           (PyArrayObject*) result, jrne)){
        Py_DECREF(result); // Free result memory
        return -1;
    }

    // Get return value data pointer and copy data into jrval
    const double *val = (double *) PyArray_DATA((PyArrayObject*) result);
    for(int i=0; i<jrne; i++) {
        jrval[i] = val[i];
    }

    // Free result memory
    Py_DECREF(result);

    return 0;
}

static int eval_h(int n, int m, int hne, const double x[], const double y[],
                  double hval[], const void *userdata){

    // Wrap input arrays as NumPy arrays
    npy_intp xdim[] = {n};
    PyArrayObject *py_x = (PyArrayObject*)
       PyArray_SimpleNewFromData(1, xdim, NPY_DOUBLE, (void *) x);
    npy_intp ydim[] = {m};
    PyArrayObject *py_y = (PyArrayObject*)
       PyArray_SimpleNewFromData(1, ydim, NPY_DOUBLE, (void *) y);

    // Build Python argument list
    PyObject *arglist = Py_BuildValue("(OO)", py_x, py_y);

    // Call Python eval_h
    PyObject *result = PyObject_CallObject(py_eval_h, arglist);
    Py_DECREF(py_x);    // Free py_x memory
    Py_DECREF(arglist); // Free arglist memory

    // Check that eval was successful
    if(!result)
        return -1;

    // Check return value is of correct type, size, and shape
    if(!check_array_double("eval_h return value",
                           (PyArrayObject*) result, hne)){
        Py_DECREF(result); // Free result memory
        return -1;
    }

    // Get return value data pointer and copy data into hval
    const double *val = (double *) PyArray_DATA((PyArrayObject*) result);
    for(int i=0; i<hne; i++) {
        hval[i] = val[i];
    }

    // Free result memory
    Py_DECREF(result);

    return 0;
}

//  *-*-*-*-*-*-*-*-*-*-   UPDATE CONTROL    -*-*-*-*-*-*-*-*-*-*

/* Update the control options: use C defaults but update any passed via Python*/
bool snls_update_control(struct snls_control_type *control,
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
        if(strcmp(key_name, "maxit") == 0){
            if(!parse_int_option(value, "maxit",
                                  &control->maxit))
                return false;
            continue;
        }
        if(strcmp(key_name, "alive_unit") == 0){
            if(!parse_int_option(value, "alive_unit",
                                  &control->alive_unit))
                return false;
            continue;
        }
        if(strcmp(key_name, "alive_file") == 0){
            if(!parse_char_option(value, "alive_file",
                                  control->alive_file,
                                  sizeof(control->alive_file)))
                return false;
            continue;
        }
        if(strcmp(key_name, "jacobian_available") == 0){
            if(!parse_int_option(value, "jacobian_available",
                                  &control->jacobian_available))
                return false;
            continue;
        }
        if(strcmp(key_name, "subproblem_solver") == 0){
            if(!parse_int_option(value, "subproblem_solver",
                                  &control->subproblem_solver))
                return false;
            continue;
        }
        if(strcmp(key_name, "non_monotone") == 0){
            if(!parse_int_option(value, "non_monotone",
                                  &control->non_monotone))
                return false;
            continue;
        }
        if(strcmp(key_name, "weight_update_strategy") == 0){
            if(!parse_int_option(value, "weight_update_strategy",
                                  &control->weight_update_strategy))
                return false;
            continue;
        }
        if(strcmp(key_name, "stop_r_absolute") == 0){
            if(!parse_double_option(value, "stop_r_absolute",
                                  &control->stop_r_absolute))
                return false;
            continue;
        }
        if(strcmp(key_name, "stop_r_relative") == 0){
            if(!parse_double_option(value, "stop_r_relative",
                                  &control->stop_r_relative))
                return false;
            continue;
        }
        if(strcmp(key_name, "stop_pg_absolute") == 0){
            if(!parse_double_option(value, "stop_pg_absolute",
                                  &control->stop_pg_absolute))
                return false;
            continue;
        }
        if(strcmp(key_name, "stop_pg_relative") == 0){
            if(!parse_double_option(value, "stop_pg_relative",
                                  &control->stop_pg_relative))
                return false;
            continue;
        }
        if(strcmp(key_name, "stop_s") == 0){
            if(!parse_double_option(value, "stop_s",
                                  &control->stop_s))
                return false;
            continue;
        }
        if(strcmp(key_name, "stop_pg_switch") == 0){
            if(!parse_double_option(value, "stop_pg_switch",
                                  &control->stop_pg_switch))
                return false;
            continue;
        }
        if(strcmp(key_name, "initial_weight") == 0){
            if(!parse_double_option(value, "initial_weight",
                                  &control->initial_weight))
                return false;
            continue;
        }
        if(strcmp(key_name, "minimum_weight") == 0){
            if(!parse_double_option(value, "minimum_weight",
                                  &control->minimum_weight))
                return false;
            continue;
        }
        if(strcmp(key_name, "eta_successful") == 0){
            if(!parse_double_option(value, "eta_successful",
                                  &control->eta_successful))
                return false;
            continue;
        }
        if(strcmp(key_name, "eta_very_successful") == 0){
            if(!parse_double_option(value, "eta_very_successful",
                                  &control->eta_very_successful))
                return false;
            continue;
        }
        if(strcmp(key_name, "eta_too_successful") == 0){
            if(!parse_double_option(value, "eta_too_successful",
                                  &control->eta_too_successful))
                return false;
            continue;
        }
        if(strcmp(key_name, "weight_decrease_min") == 0){
            if(!parse_double_option(value, "weight_decrease_min",
                                  &control->weight_decrease_min))
                return false;
            continue;
        }
        if(strcmp(key_name, "weight_decrease") == 0){
            if(!parse_double_option(value, "weight_decrease",
                                  &control->weight_decrease))
                return false;
            continue;
        }
        if(strcmp(key_name, "weight_increase") == 0){
            if(!parse_double_option(value, "weight_increase",
                                  &control->weight_increase))
                return false;
            continue;
        }
        if(strcmp(key_name, "weight_increase_max") == 0){
            if(!parse_double_option(value, "weight_increase_max",
                                  &control->weight_increase_max))
                return false;
            continue;
        }
        if(strcmp(key_name, "switch_to_newton") == 0){
            if(!parse_double_option(value, "switch_to_newton",
                                  &control->switch_to_newton))
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
        if(strcmp(key_name, "newton_acceleration") == 0){
            if(!parse_bool_option(value, "newton_acceleration",
                                  &control->newton_acceleration))
                return false;
            continue;
        }
        if(strcmp(key_name, "magic_step") == 0){
            if(!parse_bool_option(value, "magic_step",
                                  &control->magic_step))
                return false;
            continue;
        }
        if(strcmp(key_name, "print_obj") == 0){
            if(!parse_bool_option(value, "print_obj",
                                  &control->print_obj))
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
        if(strcmp(key_name, "prefix") == 0){
            if(!parse_char_option(value, "prefix",
                                  control->prefix,
                                  sizeof(control->prefix)))
                return false;
            continue;
        }
        if(strcmp(key_name, "slls_options") == 0){
            if(!slls_update_control(&control->slls_control, value))
                return false;
            continue;
        }
        if(strcmp(key_name, "sllsb_options") == 0){
            if(!sllsb_update_control(&control->sllsb_control, value))
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
// NB not static as it is used for nested options within other Python interfaces
PyObject* snls_make_options_dict(const struct snls_control_type *control){
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
    PyDict_SetItemString(py_options, "maxit",
                         PyLong_FromLong(control->maxit));
    PyDict_SetItemString(py_options, "alive_unit",
                         PyLong_FromLong(control->alive_unit));
    PyDict_SetItemString(py_options, "alive_file",
                         PyUnicode_FromString(control->alive_file));
    PyDict_SetItemString(py_options, "jacobian_available",
                         PyLong_FromLong(control->jacobian_available));
    PyDict_SetItemString(py_options, "subproblem_solver",
                         PyLong_FromLong(control->subproblem_solver));
    PyDict_SetItemString(py_options, "non_monotone",
                         PyLong_FromLong(control->non_monotone));
    PyDict_SetItemString(py_options, "weight_update_strategy",
                         PyLong_FromLong(control->weight_update_strategy));
    PyDict_SetItemString(py_options, "stop_r_absolute",
                         PyFloat_FromDouble(control->stop_r_absolute));
    PyDict_SetItemString(py_options, "stop_r_relative",
                         PyFloat_FromDouble(control->stop_r_relative));
    PyDict_SetItemString(py_options, "stop_pg_absolute",
                         PyFloat_FromDouble(control->stop_pg_absolute));
    PyDict_SetItemString(py_options, "stop_pg_relative",
                         PyFloat_FromDouble(control->stop_pg_relative));
    PyDict_SetItemString(py_options, "stop_s",
                         PyFloat_FromDouble(control->stop_s));
    PyDict_SetItemString(py_options, "stop_pg_switch",
                         PyFloat_FromDouble(control->stop_pg_switch));
    PyDict_SetItemString(py_options, "initial_weight",
                         PyFloat_FromDouble(control->initial_weight));
    PyDict_SetItemString(py_options, "minimum_weight",
                         PyFloat_FromDouble(control->minimum_weight));
    PyDict_SetItemString(py_options, "eta_successful",
                         PyFloat_FromDouble(control->eta_successful));
    PyDict_SetItemString(py_options, "eta_very_successful",
                         PyFloat_FromDouble(control->eta_very_successful));
    PyDict_SetItemString(py_options, "eta_too_successful",
                         PyFloat_FromDouble(control->eta_too_successful));
    PyDict_SetItemString(py_options, "weight_decrease_min",
                         PyFloat_FromDouble(control->weight_decrease_min));
    PyDict_SetItemString(py_options, "weight_decrease",
                         PyFloat_FromDouble(control->weight_decrease));
    PyDict_SetItemString(py_options, "weight_increase",
                         PyFloat_FromDouble(control->weight_increase));
    PyDict_SetItemString(py_options, "weight_increase_max",
                         PyFloat_FromDouble(control->weight_increase_max));
      PyDict_SetItemString(py_options, "switch_to_newton",
                         PyFloat_FromDouble(control->switch_to_newton));
    PyDict_SetItemString(py_options, "cpu_time_limit",
                         PyFloat_FromDouble(control->cpu_time_limit));
    PyDict_SetItemString(py_options, "clock_time_limit",
                         PyFloat_FromDouble(control->clock_time_limit));
    PyDict_SetItemString(py_options, "newton_acceleration",
                         PyBool_FromLong(control->newton_acceleration));
    PyDict_SetItemString(py_options, "magic_step",
                         PyBool_FromLong(control->magic_step));
    PyDict_SetItemString(py_options, "print_obj",
                         PyBool_FromLong(control->print_obj));
    PyDict_SetItemString(py_options, "space_critical",
                         PyBool_FromLong(control->space_critical));
    PyDict_SetItemString(py_options, "deallocate_error_fatal",
                         PyBool_FromLong(control->deallocate_error_fatal));
    PyDict_SetItemString(py_options, "prefix",
                         PyUnicode_FromString(control->prefix));
    PyDict_SetItemString(py_options, "slls_options",
                         slls_make_options_dict(&control->slls_control));
    PyDict_SetItemString(py_options, "sllsb_options",
                         sllsb_make_options_dict(&control->sllsb_control));

    return py_options;
}

//  *-*-*-*-*-*-*-*-*-*-   MAKE TIME    -*-*-*-*-*-*-*-*-*-*

/* Take the time struct from C and turn it into a python dictionary */
static PyObject* snls_make_time_dict(const struct snls_time_type *time){
    PyObject *py_time = PyDict_New();

    // Set float/double time entries

    PyDict_SetItemString(py_time, "total",
                         PyFloat_FromDouble(time->total));
    PyDict_SetItemString(py_time, "slls",
                         PyFloat_FromDouble(time->slls));
    PyDict_SetItemString(py_time, "sllsb",
                         PyFloat_FromDouble(time->sllsb));
    PyDict_SetItemString(py_time, "clock_total",
                         PyFloat_FromDouble(time->clock_total));
    PyDict_SetItemString(py_time, "clock_slls",
                         PyFloat_FromDouble(time->clock_slls));
    PyDict_SetItemString(py_time, "clock_sllsb",
                         PyFloat_FromDouble(time->clock_sllsb));

    return py_time;
}

//  *-*-*-*-*-*-*-*-*-*-   MAKE INFORM    -*-*-*-*-*-*-*-*-*-*

/* Take the inform struct from C and turn it into a python dictionary */
// NB not static as it is used for nested informs within other Python interfaces
PyObject* snls_make_inform_dict(const struct snls_inform_type *inform){
    PyObject *py_inform = PyDict_New();

    PyDict_SetItemString(py_inform, "status",
                         PyLong_FromLong(inform->status));
    PyDict_SetItemString(py_inform, "alloc_status",
                         PyLong_FromLong(inform->alloc_status));
    PyDict_SetItemString(py_inform, "bad_alloc",
                         PyUnicode_FromString(inform->bad_alloc));
    PyDict_SetItemString(py_inform, "bad_eval",
                         PyUnicode_FromString(inform->bad_eval));
    PyDict_SetItemString(py_inform, "iter",
                         PyLong_FromLong(inform->iter));
    PyDict_SetItemString(py_inform, "inner_iter",
                         PyLong_FromLong(inform->inner_iter));
    PyDict_SetItemString(py_inform, "r_eval",
                         PyLong_FromLong(inform->r_eval));
    PyDict_SetItemString(py_inform, "jr_eval",
                         PyLong_FromLong(inform->jr_eval));
    PyDict_SetItemString(py_inform, "obj",
                         PyFloat_FromDouble(inform->obj));
    PyDict_SetItemString(py_inform, "norm_r",
                         PyFloat_FromDouble(inform->norm_r));
    PyDict_SetItemString(py_inform, "norm_g",
                         PyFloat_FromDouble(inform->norm_g));
    PyDict_SetItemString(py_inform, "norm_pg",
                         PyFloat_FromDouble(inform->norm_pg));
    PyDict_SetItemString(py_inform, "weight",
                         PyFloat_FromDouble(inform->weight));
    PyDict_SetItemString(py_inform, "time",
                         snls_make_time_dict(&inform->time));
    PyDict_SetItemString(py_inform, "slls_inform",
                         slls_make_inform_dict(&inform->slls_inform));
    PyDict_SetItemString(py_inform, "sllsb_inform",
                         sllsb_make_inform_dict(&inform->sllsb_inform));

    return py_inform;
}

//  *-*-*-*-*-*-*-*-*-*-   SNLS_INITIALIZE    -*-*-*-*-*-*-*-*-*-*

static PyObject* py_snls_initialize(PyObject *self){

    // Call snls_initialize
    snls_initialize(&data, &control, &inform);

    // Record that SNLS has been initialised
    init_called = true;

    // Return options Python dictionary
    PyObject *py_options = snls_make_options_dict(&control);
    return Py_BuildValue("O", py_options);
}

//  *-*-*-*-*-*-*-*-*-*-*-*-   SNLS_LOAD    -*-*-*-*-*-*-*-*-*-*-*-*

static PyObject* py_snls_load(PyObject *self, PyObject *args, PyObject *keywds){
    PyArrayObject *py_Jr_row, *py_Jr_col, *py_Jr_ptr, *py_cohort;
    PyObject *py_options = NULL;
    int *Jr_row = NULL, *Jr_col = NULL, *Jr_ptr = NULL, *cohort = NULL;
    const char *Jr_type;
    int n, m_r, m_c, Jr_ne, Jr_ptr_ne;
    // Check that package has been initialised
    if(!check_init(init_called))
        return NULL;

    // Parse positional and keyword arguments
    static char *kwlist[] = {"n", "m_r", "m_c", "Jr_type", "Jr_ne", 
                             "Jr_row", "Jr_col", "Jr_ptr_ne", "Jr_ptr",
                             "cohort", "options", NULL};

    if(!PyArg_ParseTupleAndKeywords(args, keywds, "iiisiOOiO|OO",
                                    kwlist, &n, &m_r, &m_c,
                                    &Jr_type, &Jr_ne, &py_Jr_row,
                                    &py_Jr_col, &Jr_ptr_ne, &py_Jr_ptr, 
                                    &py_cohort, &py_options))
        return NULL;

    // Check that array inputs are of correct type, size, and shape

    if(!(
        check_array_int("Jr_row", py_Jr_row, Jr_ne) &&
        check_array_int("Jr_col", py_Jr_col, Jr_ne) &&
        check_array_int("Jr_ptr", py_Jr_ptr, Jr_ptr_ne) &&
        check_array_int("cohort_ptr", py_cohort, n)
        ))
        return NULL;

    // Convert 64bit integer Jr_row array to 32bit
    if((PyObject *) py_Jr_row != Py_None){
        Jr_row = malloc(Jr_ne * sizeof(int));
        long int *Jr_row_long = (long int *) PyArray_DATA(py_Jr_row);
        for(int i = 0; i < Jr_ne; i++) Jr_row[i] = (int) Jr_row_long[i];
    }

    // Convert 64bit integer Jr_col array to 32bit
    if((PyObject *) py_Jr_col != Py_None){
        Jr_col = malloc(Jr_ne * sizeof(int));
        long int *Jr_col_long = (long int *) PyArray_DATA(py_Jr_col);
        for(int i = 0; i < Jr_ne; i++) Jr_col[i] = (int) Jr_col_long[i];
    }

    // Convert 64bit integer Jr_ptr array to 32bit
    if((PyObject *) py_Jr_ptr != Py_None){
        Jr_ptr = malloc((n+1) * sizeof(int));
        long int *Jr_ptr_long = (long int *) PyArray_DATA(py_Jr_ptr);
        for(int i = 0; i < n+1; i++) Jr_ptr[i] = (int) Jr_ptr_long[i];
    }

    // Convert 64bit integer cohort array to 32bit
    if((PyObject *) py_cohort != Py_None){
        cohort = malloc(n * sizeof(int));
        long int *cohort_long = (long int *) PyArray_DATA(py_cohort);
        for(int i = 0; i < n; i++) cohort[i] = (int) cohort_long[i];
    }

    // Reset control options
    snls_reset_control(&control, &data, &status);

    // Update SNLS control options
    if(!snls_update_control(&control, py_options))
        return NULL;

    // Call snls_import
    snls_import(&control, &data, &status, n, m_r, m_c,
                Jr_type, Jr_ne, Jr_row, Jr_col, Jr_ptr_ne, Jr_ptr, cohort);

    // Free allocated memory
    if(Jr_row != NULL) free(Jr_row);
    if(Jr_col != NULL) free(Jr_col);
    if(Jr_ptr != NULL) free(Jr_ptr);
    if(cohort != NULL) free(cohort);

    // Raise any status errors
    if(!check_error_codes(status))
        return NULL;

    // Return None boilerplate
    Py_INCREF(Py_None);
    return Py_None;
}

//  *-*-*-*-*-*-*-*-*-*-   SNLS_SOLVE   -*-*-*-*-*-*-*-*

static PyObject* py_snls_solve(PyObject *self, PyObject *args, PyObject *keywds){
    PyArrayObject *py_x, *py_w = NULL;
    PyObject *temp_r, *temp_jr;
    double *x, *w = NULL;
    int n, m_r, m_c, Jr_ne;

    // Check that package has been initialised
    if(!check_init(init_called))
        return NULL;

    // Parse positional arguments
    static char *kwlist[] = {"n", "m_r", "m_c", "x", "eval_r", 
                             "Jr_ne", "eval_jr", "w", NULL};
    if(!PyArg_ParseTupleAndKeywords(args, keywds, "iiiOOiO|O", kwlist, 
                                    &n, &m_r, &m_c, &py_x, &temp_r, 
                                    &Jr_ne, &temp_jr, &py_w))
        return NULL;

    // Check that array inputs are of correct type, size, and shape
    if(!check_array_double("x", py_x, n))
        return NULL;
    if(py_w != NULL) {
      if((PyObject *) py_w != Py_None){
        if(!check_array_double("w", py_w, m_r))
            return NULL;
        w = (double *) PyArray_DATA(py_w);
      }  
    }

    // Get array data pointer
    x = (double *) PyArray_DATA(py_x);

    // Check that functions are callable
    if(!(
        check_callable(temp_r) &&
        check_callable(temp_jr)
        ))
        return NULL;

    // Store functions
    Py_XINCREF(temp_r);            /* Add a reference to new callback */
    Py_XDECREF(py_eval_r);         /* Dispose of previous callback */
    py_eval_r = temp_r;            /* Remember new callback */
    Py_XINCREF(temp_jr);            /* Add a reference to new callback */
    Py_XDECREF(py_eval_jr);        /* Dispose of previous callback */
    py_eval_jr = temp_jr;          /* Remember new callback */

  // Create NumPy output arrays
    npy_intp ndim[] = {n}; // size of z, g and x_stat
    npy_intp mrdim[] = {m_r}; // size of r
    npy_intp mcdim[] = {m_c}; // size of y

    PyArrayObject *py_y =
      (PyArrayObject *) PyArray_SimpleNew(1, mcdim, NPY_DOUBLE);
    double *y = (double *) PyArray_DATA(py_y);
    PyArrayObject *py_z =
      (PyArrayObject *) PyArray_SimpleNew(1, ndim, NPY_DOUBLE);
    double *z = (double *) PyArray_DATA(py_z);
    PyArrayObject *py_r =
      (PyArrayObject *) PyArray_SimpleNew(1, mrdim, NPY_DOUBLE);
    double *r = (double *) PyArray_DATA(py_r);
    PyArrayObject *py_g =
      (PyArrayObject *) PyArray_SimpleNew(1, ndim, NPY_DOUBLE);
    double *g = (double *) PyArray_DATA(py_g);
    PyArrayObject *py_x_stat =
      (PyArrayObject *) PyArray_SimpleNew(1, ndim, NPY_INT);
    int *x_stat = (int *) PyArray_DATA(py_x_stat);

    // Call snls_solve_direct
    status = 1; // set status to 1 on entry
    snls_solve_with_jac(&data, NULL, &status, n, m_r, m_c, 
                        x, y, z, r, g, x_stat, 
                        eval_r, Jr_ne, eval_jr, w );

    // Propagate any errors with the callback function
    if(PyErr_Occurred())
        return NULL;

    // Raise any status errors
    if(!check_error_codes(status))
        return NULL;

   // Return x, y, z, r, g and x_stat
    snls_solve_return = Py_BuildValue("OOOOOO", py_x, py_y, py_z, py_r, py_g,
                                      py_x_stat);
    Py_INCREF(snls_solve_return);
    return snls_solve_return;

}

//  *-*-*-*-*-*-*-*-*-*-   SNLS_INFORMATION   -*-*-*-*-*-*-*-*

static PyObject* py_snls_information(PyObject *self){

    // Check that package has been initialised
    if(!check_init(init_called))
        return NULL;

    // Call snls_information
    snls_information(&data, &inform, &status);

    // Return status and inform Python dictionary
    PyObject *py_inform = snls_make_inform_dict(&inform);
    return Py_BuildValue("O", py_inform);
}

//  *-*-*-*-*-*-*-*-*-*-   SNLS_TERMINATE   -*-*-*-*-*-*-*-*-*-*

static PyObject* py_snls_terminate(PyObject *self){

    // Check that package has been initialised
    if(!check_init(init_called))
        return NULL;

    // Call snls_terminate
    snls_terminate(&data, &control, &inform);

    // Return None boilerplate
    Py_INCREF(Py_None);
    return Py_None;
}

//  *-*-*-*-*-*-*-*-*-*-   INITIALIZE SNLS PYTHON MODULE    -*-*-*-*-*-*-*-*-*-*

/* snls python module method table */
static PyMethodDef snls_module_methods[] = {
    {"initialize", (PyCFunction) py_snls_initialize, METH_NOARGS, NULL},
    {"load", (PyCFunction) py_snls_load, METH_VARARGS | METH_KEYWORDS, NULL},
    {"solve", (PyCFunction) py_snls_solve, METH_VARARGS | METH_KEYWORDS, NULL},
    {"information", (PyCFunction) py_snls_information, METH_NOARGS, NULL},
    {"terminate", (PyCFunction) py_snls_terminate, METH_NOARGS, NULL},
    {NULL, NULL, 0, NULL}  /* Sentinel */
};

/* snls python module documentation */

PyDoc_STRVAR(snls_module_doc,

"The snls package uses a regularization method to find a (local) unconstrained\n"
"minimizer of a differentiable weighted sum-of-squares objective function\n"
"f(x) :=\n"
"   1/2 sum_{i=1}^m w_i c_i^2(x) == 1/2 ||c(x)||^2_W\n"
"of many variables f{x} involving positive weights w_i, i=1,...,m.\n"
"The method offers the choice of direct and iterative solution of the key\n"
"regularization subproblems, and is most suitable for large problems.\n"
"First derivatives of the residual function c(x) are required, and if\n"
"second derivatives of the c_i(x) can be calculated, they may be exploited.\n"
"\n"
"See $GALAHAD/html/Python/snls.html for argument lists, call order\n"
"and other details.\n"
"\n"
);

/* snls python module definition */
static struct PyModuleDef module = {
   PyModuleDef_HEAD_INIT,
   "snls",               /* name of module */
   snls_module_doc,      /* module documentation, may be NULL */
   -1,                  /* size of per-interpreter state of the module,or -1
                           if the module keeps state in global variables */
   snls_module_methods   /* module methods */
};

/* Python module initialization */
PyMODINIT_FUNC PyInit_snls(void) { // must be same as module name above
    import_array();  // for NumPy arrays
    return PyModule_Create(&module);
}
