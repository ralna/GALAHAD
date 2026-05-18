//* \file rqs_pyiface.c */

/*
 * THIS VERSION: GALAHAD 5.0 - 2024-06-15 AT 11:30 GMT.
 *
 *-*-*-*-*-*-*-*-*-  GALAHAD_RQS PYTHON INTERFACE  *-*-*-*-*-*-*-*-*-*-
 *
 *  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
 *  Principal author: Jaroslav Fowkes & Nick Gould
 *
 *  History -
 *   originally released GALAHAD Version 4.1. April 13th 2023
 *
 *  For full documentation, see
 *   http://galahad.rl.ac.uk/galahad-www/specs.html
 */

#include "galahad_python.h"
#include "galahad_rqs.h"

/* Nested SLS and IR control and inform prototypes */
bool sls_update_control(struct sls_control_type *control,
                        PyObject *py_options);
PyObject* sls_make_options_dict(const struct sls_control_type *control);
PyObject* sls_make_inform_dict(const struct sls_inform_type *inform);
bool ir_update_control(struct ir_control_type *control,
                        PyObject *py_options);
PyObject* ir_make_options_dict(const struct ir_control_type *control);
PyObject* ir_make_inform_dict(const struct ir_inform_type *inform);

/* Module global variables */
static void *data;                       // private internal data
static struct rqs_control_type control;  // control struct
static struct rqs_inform_type inform;    // inform struct
static bool init_called = false;         // record if initialise was called
static bool load_called = false;         // record if load was called
static bool load_m_called = false;       // record if load_m was called
static bool load_a_called = false;       // record if load_a was called
static int status = 0;                   // exit status

//  *-*-*-*-*-*-*-*-*-*-   UPDATE CONTROL    -*-*-*-*-*-*-*-*-*-*

/* Update the control options: use C defaults but update any passed via Python*/
// NB not static as it is used for nested control within ARC Python interface
bool rqs_update_control(struct rqs_control_type *control,
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
        if(strcmp(key_name, "problem") == 0){
            if(!parse_int_option(value, "problem",
                                  &control->problem))
                return false;
            continue;
        }
        if(strcmp(key_name, "print_level") == 0){
            if(!parse_int_option(value, "print_level",
                                  &control->print_level))
                return false;
            continue;
        }
        if(strcmp(key_name, "dense_factorization") == 0){
            if(!parse_int_option(value, "dense_factorization",
                                  &control->dense_factorization))
                return false;
            continue;
        }
        if(strcmp(key_name, "new_h") == 0){
            if(!parse_int_option(value, "new_h",
                                  &control->new_h))
                return false;
            continue;
        }
        if(strcmp(key_name, "new_m") == 0){
            if(!parse_int_option(value, "new_m",
                                  &control->new_m))
                return false;
            continue;
        }
        if(strcmp(key_name, "new_a") == 0){
            if(!parse_int_option(value, "new_a",
                                  &control->new_a))
                return false;
            continue;
        }
        if(strcmp(key_name, "max_factorizations") == 0){
            if(!parse_int_option(value, "max_factorizations",
                                  &control->max_factorizations))
                return false;
            continue;
        }
        if(strcmp(key_name, "inverse_itmax") == 0){
            if(!parse_int_option(value, "inverse_itmax",
                                  &control->inverse_itmax))
                return false;
            continue;
        }
        if(strcmp(key_name, "taylor_max_degree") == 0){
            if(!parse_int_option(value, "taylor_max_degree",
                                  &control->taylor_max_degree))
                return false;
            continue;
        }
        if(strcmp(key_name, "initial_multiplier") == 0){
            if(!parse_double_option(value, "initial_multiplier",
                                  &control->initial_multiplier))
                return false;
            continue;
        }
        if(strcmp(key_name, "lower") == 0){
            if(!parse_double_option(value, "lower",
                                  &control->lower))
                return false;
            continue;
        }
        if(strcmp(key_name, "upper") == 0){
            if(!parse_double_option(value, "upper",
                                  &control->upper))
                return false;
            continue;
        }
        if(strcmp(key_name, "stop_normal") == 0){
            if(!parse_double_option(value, "stop_normal",
                                  &control->stop_normal))
                return false;
            continue;
        }
        if(strcmp(key_name, "stop_hard") == 0){
            if(!parse_double_option(value, "stop_hard",
                                  &control->stop_hard))
                return false;
            continue;
        }
        if(strcmp(key_name, "start_invit_tol") == 0){
            if(!parse_double_option(value, "start_invit_tol",
                                  &control->start_invit_tol))
                return false;
            continue;
        }
        if(strcmp(key_name, "start_invitmax_tol") == 0){
            if(!parse_double_option(value, "start_invitmax_tol",
                                  &control->start_invitmax_tol))
                return false;
            continue;
        }
        if(strcmp(key_name, "use_initial_multiplier") == 0){
            if(!parse_bool_option(value, "use_initial_multiplier",
                                  &control->use_initial_multiplier))
                return false;
            continue;
        }
        if(strcmp(key_name, "initialize_approx_eigenvector") == 0){
            if(!parse_bool_option(value, "initialize_approx_eigenvector",
                                  &control->initialize_approx_eigenvector))
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
        if(strcmp(key_name, "problem_file") == 0){
            if(!parse_char_option(value, "problem_file",
                                  control->problem_file,
                                  sizeof(control->problem_file)))
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
        if(strcmp(key_name, "prefix") == 0){
            if(!parse_char_option(value, "prefix",
                                  control->prefix,
                                  sizeof(control->prefix)))
                return false;
            continue;
        }
        if(strcmp(key_name, "sls_options") == 0){
            if(!sls_update_control(&control->sls_control, value))
                return false;
            continue;
        }
        if(strcmp(key_name, "ir_options") == 0){
            if(!ir_update_control(&control->ir_control, value))
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
// NB not static as it is used for nested inform within ARC Python interface
PyObject* rqs_make_options_dict(const struct rqs_control_type *control){
    PyObject *py_options = PyDict_New();

    PyDict_SetItemString(py_options, "error",
                         PyLong_FromLong(control->error));
    PyDict_SetItemString(py_options, "out",
                         PyLong_FromLong(control->out));
    PyDict_SetItemString(py_options, "problem",
                         PyLong_FromLong(control->problem));
    PyDict_SetItemString(py_options, "print_level",
                         PyLong_FromLong(control->print_level));
    PyDict_SetItemString(py_options, "dense_factorization",
                         PyLong_FromLong(control->dense_factorization));
    PyDict_SetItemString(py_options, "new_h",
                         PyLong_FromLong(control->new_h));
    PyDict_SetItemString(py_options, "new_m",
                         PyLong_FromLong(control->new_m));
    PyDict_SetItemString(py_options, "new_a",
                         PyLong_FromLong(control->new_a));
    PyDict_SetItemString(py_options, "max_factorizations",
                         PyLong_FromLong(control->max_factorizations));
    PyDict_SetItemString(py_options, "inverse_itmax",
                         PyLong_FromLong(control->inverse_itmax));
    PyDict_SetItemString(py_options, "taylor_max_degree",
                         PyLong_FromLong(control->taylor_max_degree));
    PyDict_SetItemString(py_options, "initial_multiplier",
                         PyFloat_FromDouble(control->initial_multiplier));
    PyDict_SetItemString(py_options, "lower",
                         PyFloat_FromDouble(control->lower));
    PyDict_SetItemString(py_options, "upper",
                         PyFloat_FromDouble(control->upper));
    PyDict_SetItemString(py_options, "stop_normal",
                         PyFloat_FromDouble(control->stop_normal));
    PyDict_SetItemString(py_options, "stop_hard",
                         PyFloat_FromDouble(control->stop_hard));
    PyDict_SetItemString(py_options, "start_invit_tol",
                         PyFloat_FromDouble(control->start_invit_tol));
    PyDict_SetItemString(py_options, "start_invitmax_tol",
                         PyFloat_FromDouble(control->start_invitmax_tol));
    PyDict_SetItemString(py_options, "use_initial_multiplier",
                         PyBool_FromLong(control->use_initial_multiplier));
    PyDict_SetItemString(py_options, "initialize_approx_eigenvector",
                         PyBool_FromLong(control->initialize_approx_eigenvector));
    PyDict_SetItemString(py_options, "space_critical",
                         PyBool_FromLong(control->space_critical));
    PyDict_SetItemString(py_options, "deallocate_error_fatal",
                         PyBool_FromLong(control->deallocate_error_fatal));
    PyDict_SetItemString(py_options, "problem_file",
                         PyUnicode_FromString(control->problem_file));
    PyDict_SetItemString(py_options, "symmetric_linear_solver",
                         PyUnicode_FromString(control->symmetric_linear_solver));
    PyDict_SetItemString(py_options, "definite_linear_solver",
                         PyUnicode_FromString(control->definite_linear_solver));
    PyDict_SetItemString(py_options, "prefix",
                         PyUnicode_FromString(control->prefix));
    PyDict_SetItemString(py_options, "sls_options",
                         sls_make_options_dict(&control->sls_control));
    PyDict_SetItemString(py_options, "ir_options",
                         ir_make_options_dict(&control->ir_control));

    return py_options;
}

//  *-*-*-*-*-*-*-*-*-*-   MAKE TIME    -*-*-*-*-*-*-*-*-*-*

/* Take the time struct from C and turn it into a python dictionary */
static PyObject* rqs_make_time_dict(const struct rqs_time_type *time){
    PyObject *py_time = PyDict_New();

    // Set float/double time entries

    PyDict_SetItemString(py_time, "total",
                         PyFloat_FromDouble(time->total));
    PyDict_SetItemString(py_time, "assemble",
                         PyFloat_FromDouble(time->assemble));
    PyDict_SetItemString(py_time, "analyse",
                         PyFloat_FromDouble(time->analyse));
    PyDict_SetItemString(py_time, "factorize",
                         PyFloat_FromDouble(time->factorize));
    PyDict_SetItemString(py_time, "solve",
                         PyFloat_FromDouble(time->solve));
    PyDict_SetItemString(py_time, "clock_total",
                         PyFloat_FromDouble(time->clock_total));
    PyDict_SetItemString(py_time, "clock_assemble",
                         PyFloat_FromDouble(time->clock_assemble));
    PyDict_SetItemString(py_time, "clock_analyse",
                         PyFloat_FromDouble(time->clock_analyse));
    PyDict_SetItemString(py_time, "clock_factorize",
                         PyFloat_FromDouble(time->clock_factorize));
    PyDict_SetItemString(py_time, "clock_solve",
                         PyFloat_FromDouble(time->clock_solve));
    return py_time;
}

//  *-*-*-*-*-*-*-*-*-*-   MAKE HISTORY    -*-*-*-*-*-*-*-*-*-*

/* Take the time struct from C and turn it into a python dictionary */
static PyObject* rqs_make_history_dict(const struct rqs_history_type *history){
    PyObject *py_history = PyDict_New();

    PyDict_SetItemString(py_history, "lambda",
                         PyFloat_FromDouble(history->lambda));
    PyDict_SetItemString(py_history, "x_norm",
                         PyFloat_FromDouble(history->x_norm));

    return py_history;
}

//  *-*-*-*-*-*-*-*-*-*-   MAKE INFORM    -*-*-*-*-*-*-*-*-*-*

/* Take the inform struct from C and turn it into a python dictionary */
// NB not static as it is used for nested control within ARC Python interface
PyObject* rqs_make_inform_dict(const struct rqs_inform_type *inform){
    PyObject *py_inform = PyDict_New();

    PyDict_SetItemString(py_inform, "status",
                         PyLong_FromLong(inform->status));
    PyDict_SetItemString(py_inform, "alloc_status",
                         PyLong_FromLong(inform->alloc_status));
    PyDict_SetItemString(py_inform, "factorizations",
                         PyLong_FromLong(inform->factorizations));
    PyDict_SetItemString(py_inform, "max_entries_factors",
                         PyLong_FromLong(inform->max_entries_factors));
    PyDict_SetItemString(py_inform, "len_history",
                         PyLong_FromLong(inform->len_history));
    PyDict_SetItemString(py_inform, "obj",
                         PyFloat_FromDouble(inform->obj));
    PyDict_SetItemString(py_inform, "obj_regularized",
                         PyFloat_FromDouble(inform->obj_regularized));
    PyDict_SetItemString(py_inform, "x_norm",
                         PyFloat_FromDouble(inform->x_norm));
    PyDict_SetItemString(py_inform, "multiplier",
                         PyFloat_FromDouble(inform->multiplier));
    PyDict_SetItemString(py_inform, "pole",
                         PyFloat_FromDouble(inform->pole));
    PyDict_SetItemString(py_inform, "dense_factorization",
                         PyBool_FromLong(inform->dense_factorization));
    PyDict_SetItemString(py_inform, "hard_case",
                         PyBool_FromLong(inform->hard_case));
    PyDict_SetItemString(py_inform, "bad_alloc",
                         PyUnicode_FromString(inform->bad_alloc));

    // include history arrays
    //npy_intp hdim[] = {1000};
    //PyArrayObject *py_lambda =
    //  (PyArrayObject*) PyArray_SimpleNew(1, hdim, NPY_DOUBLE);
    //double *lambda = (double *) PyArray_DATA(py_lambda);
    //for(int i=0; i<1000; i++) lambda[i] = inform->history[i]->lambda;
    //PyDict_SetItemString(py_inform, "lambda", (PyObject *) py_lambda);
    //PyArrayObject *py_x_norm =
    //  (PyArrayObject*) PyArray_SimpleNew(1, hdim, NPY_DOUBLE);
    //double *x_norm = (double *) PyArray_DATA(py_x_norm);
    //for(int i=0; i<1000; i++) x_norm[i] = inform->history[i]->x_norm;
    //PyDict_SetItemString(py_inform, "x_norm", (PyObject *) py_x_norm);

    // Set time nested dictionary
    PyDict_SetItemString(py_inform, "time",
                         rqs_make_time_dict(&inform->time));

    // Set dictionaries from subservient packages
    PyDict_SetItemString(py_inform, "sls_inform",
                         sls_make_inform_dict(&inform->sls_inform));
    PyDict_SetItemString(py_inform, "ir_inform",
                         ir_make_inform_dict(&inform->ir_inform));

    return py_inform;
}

//  *-*-*-*-*-*-*-*-*-*-   RQS_INITIALIZE    -*-*-*-*-*-*-*-*-*-*

static PyObject* py_rqs_initialize(PyObject *self){

    // Call rqs_initialize
    rqs_initialize(&data, &control, &status);

    // Record that rqs has been initialised
    init_called = true;

    // Return options Python dictionary
    PyObject *py_options = rqs_make_options_dict(&control);
    return Py_BuildValue("O", py_options);
}

//  *-*-*-*-*-*-*-*-*-*-*-*-   RQS_LOAD    -*-*-*-*-*-*-*-*-*-*-*-*

static PyObject* py_rqs_load(PyObject *self, PyObject *args, PyObject *keywds){
    PyArrayObject *py_H_row, *py_H_col, *py_H_ptr;
    PyObject *py_options = NULL;
    int *H_row = NULL, *H_col = NULL, *H_ptr = NULL;
    const char *H_type;
    int n, H_ne;

    // Check that package has been initialised
    if(!check_init(init_called))
        return NULL;

    // Parse positional and keyword arguments
    static char *kwlist[] = {"n","H_type","H_ne","H_row","H_col","H_ptr",
                             "options",NULL};

    if(!PyArg_ParseTupleAndKeywords(args, keywds, "isiOOO|O", kwlist, &n,
                                    &H_type, &H_ne, &py_H_row,
                                    &py_H_col, &py_H_ptr,
                                    &py_options))
        return NULL;

    // Check that array inputs are of correct type, size, and shape

    if(!(
        check_array_int("H_row", py_H_row, H_ne) &&
        check_array_int("H_col", py_H_col, H_ne) &&
        check_array_int("H_ptr", py_H_ptr, n+1)
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

    // Reset control options
    rqs_reset_control(&control, &data, &status);

    // Update rqs control options
    if(!rqs_update_control(&control, py_options))
        return NULL;

    // Call rqs_import
    rqs_import(&control, &data, &status, n,
               H_type, H_ne, H_row, H_col, H_ptr);

    // Free allocated memory
    if(H_row != NULL) free(H_row);
    if(H_col != NULL) free(H_col);
    if(H_ptr != NULL) free(H_ptr);

    // Raise any status errors
    if(!check_error_codes(status))
        return NULL;

    // Record that rqs structures been initialised
    load_called = true;

    // Return None boilerplate
    Py_INCREF(Py_None);
    return Py_None;
}

//  *-*-*-*-*-*-*-*-*-*-*-*-   RQS_LOAD_M    -*-*-*-*-*-*-*-*-*-*-*-*

static PyObject* py_rqs_load_m(PyObject *self, PyObject *args,
                               PyObject *keywds){
    PyArrayObject *py_M_row, *py_M_col, *py_M_ptr;
    PyObject *py_options = NULL;
    int *M_row = NULL, *M_col = NULL, *M_ptr = NULL;
    const char *M_type;
    int n, M_ne;

    // Check that package has been initialised
    if(!check_init(init_called))
        return NULL;

    // Parse positional and keyword arguments
    static char *kwlist[] = {"n","M_type","M_ne","M_row","M_col","M_ptr",
                             "options",NULL};

    if(!PyArg_ParseTupleAndKeywords(args, keywds, "isiOOO|O", kwlist, &n,
                                    &M_type, &M_ne, &py_M_row,
                                    &py_M_col, &py_M_ptr,
                                    &py_options))
        return NULL;

    // Check that array inputs are of correct type, size, and shape

    if(!(
        check_array_int("M_row", py_M_row, M_ne) &&
        check_array_int("M_col", py_M_col, M_ne) &&
        check_array_int("M_ptr", py_M_ptr, n+1)
        ))
        return NULL;

    // Convert 64bit integer M_row array to 32bit
    if((PyObject *) py_M_row != Py_None){
        M_row = malloc(M_ne * sizeof(int));
        long int *M_row_long = (long int *) PyArray_DATA(py_M_row);
        for(int i = 0; i < M_ne; i++) M_row[i] = (int) M_row_long[i];
    }

    // Convert 64bit integer M_col array to 32bit
    if((PyObject *) py_M_col != Py_None){
        M_col = malloc(M_ne * sizeof(int));
        long int *M_col_long = (long int *) PyArray_DATA(py_M_col);
        for(int i = 0; i < M_ne; i++) M_col[i] = (int) M_col_long[i];
    }

    // Convert 64bit integer M_ptr array to 32bit
    if((PyObject *) py_M_ptr != Py_None){
        M_ptr = malloc((n+1) * sizeof(int));
        long int *M_ptr_long = (long int *) PyArray_DATA(py_M_ptr);
        for(int i = 0; i < n+1; i++) M_ptr[i] = (int) M_ptr_long[i];
    }

    // Reset control options
    rqs_reset_control(&control, &data, &status);

    // Update rqs control options
    if(!rqs_update_control(&control, py_options))
        return NULL;

    // Call rqs_import
    rqs_import_m(&data, &status, n,
                 M_type, M_ne, M_row, M_col, M_ptr);

    // Free allocated memory
    if(M_row != NULL) free(M_row);
    if(M_col != NULL) free(M_col);
    if(M_ptr != NULL) free(M_ptr);

    // Raise any status errors
    if(!check_error_codes(status))
        return NULL;

    // Record that rqs M structure been initialised
    load_m_called = true;

    // Return None boilerplate
    Py_INCREF(Py_None);
    return Py_None;
}

//  *-*-*-*-*-*-*-*-*-*-*-*-   RQS_LOAD_A    -*-*-*-*-*-*-*-*-*-*-*-*

static PyObject* py_rqs_load_a(PyObject *self, PyObject *args,
                               PyObject *keywds){
    PyArrayObject *py_A_row, *py_A_col, *py_A_ptr;
    PyObject *py_options = NULL;
    int *A_row = NULL, *A_col = NULL, *A_ptr = NULL;
    const char *A_type;
    int m, A_ne;

    // Check that package has been initialised
    if(!check_init(init_called))
        return NULL;

    // Parse positional and keyword arguments
    static char *kwlist[] = {"m","A_type","A_ne","A_row","A_col","A_ptr",
                             "options",NULL};

    if(!PyArg_ParseTupleAndKeywords(args, keywds, "isiOOO|O", kwlist, &m,
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
    rqs_reset_control(&control, &data, &status);

    // Update rqs control options
    if(!rqs_update_control(&control, py_options))
        return NULL;

    // Call rqs_import
    rqs_import_a(&data, &status, m,
                 A_type, A_ne, A_row, A_col, A_ptr);

    // Free allocated memory
    if(A_row != NULL) free(A_row);
    if(A_col != NULL) free(A_col);
    if(A_ptr != NULL) free(A_ptr);

    // Raise any status errors
    if(!check_error_codes(status))
        return NULL;

    // Record that rqs A structure been initialised
    load_a_called = true;

    // Return None boilerplate
    Py_INCREF(Py_None);
    return Py_None;
}

//  *-*-*-*-*-*-*-*-*-*-   RQS_SOLVE_PROBLEM   -*-*-*-*-*-*-*-*

static PyObject* py_rqs_solve_problem(PyObject *self, PyObject *args,
                                      PyObject *keywds){
    PyArrayObject *py_g, *py_H_val, *py_M_val = NULL, *py_A_val = NULL;
    double *g, *H_val, *M_val = NULL, *A_val = NULL;
    int n, m, H_ne, M_ne, A_ne;
    double power, weight, f;

    // Check that package has been initialised
    if(!check_load(load_called))
        return NULL;

    // Parse positional arguments
    static char *kwlist[] = {"n","power","weight","f","g","H_ne","H_val",
                             "M_ne","M_val","m","A_ne","A_val",NULL};

    if(!PyArg_ParseTupleAndKeywords(args, keywds, "idddOiO|iOiiO", kwlist,
                                    &n, &power, &weight, &f, &py_g,
                                    &H_ne, &py_H_val, &M_ne, &py_M_val,
                                    &m, &A_ne, &py_A_val))
        return NULL;

    // Check that array inputs are of correct type, size, and shape
    if(!check_array_double("g", py_g, n))
        return NULL;
    if(!check_array_double("H_val", py_H_val, H_ne))
        return NULL;
    if(load_m_called) {
      if(!check_array_double("M_val", py_M_val, M_ne))
          return NULL;
    }
    if(load_a_called) {
      if(!check_array_double("A_val", py_A_val, A_ne))
          return NULL;
    } else {
      m = 0;
    }

    // Get array data pointer
    g = (double *) PyArray_DATA(py_g);
    H_val = (double *) PyArray_DATA(py_H_val);
    if(py_M_val != NULL) M_val = (double *) PyArray_DATA(py_M_val);
    if(py_A_val != NULL) A_val = (double *) PyArray_DATA(py_A_val);

   // Create NumPy output arrays
    npy_intp ndim[] = {n}; // size of x
    npy_intp mdim[] = {m}; // size of y
    PyArrayObject *py_x =
      (PyArrayObject *) PyArray_SimpleNew(1, ndim, NPY_DOUBLE);
    double *x = (double *) PyArray_DATA(py_x);
    PyArrayObject *py_y =
      (PyArrayObject *) PyArray_SimpleNew(1, mdim, NPY_DOUBLE);
    double *y = (double *) PyArray_DATA(py_y);

    // Call rqs_solve_problem
    rqs_solve_problem(&data, &status, n, power, weight, f, g, H_ne, H_val, x,
                      M_ne, M_val, m, A_ne, A_val, y);
    // for( int i = 0; i < n; i++) printf("x %f\n", x[i]);
    // for( int i = 0; i < m; i++) printf("y %f\n", y[i]);

    // Propagate any errors with the callback function
    if(PyErr_Occurred())
        return NULL;

    // Raise any status errors
    if(!check_error_codes(status))
        return NULL;

    // Return x and possibly y
    PyObject *solve_problem_return;

    if(load_a_called) {
      solve_problem_return = Py_BuildValue("OO", py_x, py_y);
    } else {
      solve_problem_return = Py_BuildValue("O", py_x);
    }
    Py_INCREF(solve_problem_return);
    return solve_problem_return;

}

//  *-*-*-*-*-*-*-*-*-*-   RQS_INFORMATION   -*-*-*-*-*-*-*-*

static PyObject* py_rqs_information(PyObject *self){

    // Check that package has been initialised
    if(!check_init(init_called))
        return NULL;

    // Call rqs_information
    rqs_information(&data, &inform, &status);

    // Return status and inform Python dictionary
    PyObject *py_inform = rqs_make_inform_dict(&inform);
    return Py_BuildValue("O", py_inform);
}

//  *-*-*-*-*-*-*-*-*-*-   RQS_TERMINATE   -*-*-*-*-*-*-*-*-*-*

static PyObject* py_rqs_terminate(PyObject *self){

    // Check that package has been initialised
    if(!check_init(init_called))
        return NULL;

    // Call rqs_terminate
    rqs_terminate(&data, &control, &inform);

    // require future calls start with rqs_initialize
    init_called = false;
    load_called = false;
    load_m_called = false;
    load_a_called = false;

    // Return None boilerplate
    Py_INCREF(Py_None);
    return Py_None;
}

//  *-*-*-*-*-*-*-*-*-*-   INITIALIZE RQS PYTHON MODULE    -*-*-*-*-*-*-*-*-*-*

/* rqs python module method table */
static PyMethodDef rqs_module_methods[] = {
    {"initialize", (PyCFunction) py_rqs_initialize, METH_NOARGS,NULL},
    {"load", (PyCFunction) py_rqs_load, METH_VARARGS | METH_KEYWORDS, NULL},
    {"load_m", (PyCFunction) py_rqs_load_m, METH_VARARGS | METH_KEYWORDS, NULL},
    {"load_a", (PyCFunction) py_rqs_load_a, METH_VARARGS | METH_KEYWORDS, NULL},
    {"solve_problem", (PyCFunction) py_rqs_solve_problem, METH_VARARGS | METH_KEYWORDS, NULL},
    {"information", (PyCFunction) py_rqs_information, METH_NOARGS, NULL},
    {"terminate", (PyCFunction) py_rqs_terminate, METH_NOARGS, NULL},
    {NULL, NULL, 0, NULL}  /* Sentinel */
};

/* rqs python module documentation */

PyDoc_STRVAR(rqs_module_doc,

"The rqs package uses matrix factorization to find the global minimizer \n"
"of a regularized quadratic objective function.\n"
"The aim is to minimize the regularized quadratic objective function\n"
"r(x) = f + g^T x + 1/2 x^T H x + sigma / p ||x||_M^p, \n"
"where the weight sigma > 0, the power p >= 2, the  vector x\n"
"may optionally  be required to satisfy affine constraints A x = 0,\n"
"and where the M-norm of x is defined to be ||x||_M = sqrt{x^T M x}.\n"
"\n"
"The matrix M need not be provided in the commonly-occurring\n"
"l_2-regularization case for which M = I, the n by n identity matrix.\n"
"\n"
"Factorization of matrices of the form H + lambda M, or\n"
"( H + lambda M  A^T )\n"
"(      A         0  )\n"
"in cases where A x = 0 is imposed, for a succession\n"
"of scalars lambda will be required, so this package is most suited\n"
"for the case where such a factorization may be found efficiently.\n"
"If this is not the case, the package gltr may be preferred.\n"
"\n"
"See $GALAHAD/html/Python/rqs.html for argument lists, call order\n"
"and other details.\n"
"\n"
);

/* rqs python module definition */
static struct PyModuleDef module = {
   PyModuleDef_HEAD_INIT,
   "rqs",               /* name of module */
   rqs_module_doc,      /* module documentation, may be NULL */
   -1,                  /* size of per-interpreter state of the module,or -1
                           if the module keeps state in global variables */
   rqs_module_methods   /* module methods */
};

/* Python module initialization */
PyMODINIT_FUNC PyInit_rqs(void) { // must be same as module name above
    import_array();  // for NumPy arrays
    return PyModule_Create(&module);
}
