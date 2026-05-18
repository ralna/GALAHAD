//* \file nrek_pyiface.c */

/*
 * THIS VERSION: GALAHAD 5.4 - 2025-11-21 AT 15:40 GMT.
 *
 *-*-*-*-*-*-*-*-*-  GALAHAD_NREK PYTHON INTERFACE  *-*-*-*-*-*-*-*-*-*-
 *
 *  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
 *  Principal authors: Hussam Al Daas, Jaroslav Fowkes & Nick Gould
 *
 *  History -
 *   originally released GALAHAD Version 5.4. November 16th 2025
 *
 *  For full documentation, see
 *   http://galahad.rl.ac.uk/galahad-www/specs.html
 */

#include "galahad_python.h"
#include "galahad_nrek.h"

/* Nested SLS and RQS control and inform prototypes */
bool sls_update_control(struct sls_control_type *control,
                        PyObject *py_options);
PyObject* sls_make_options_dict(const struct sls_control_type *control);
PyObject* sls_make_inform_dict(const struct sls_inform_type *inform);
bool rqs_update_control(struct rqs_control_type *control,
                       PyObject *py_options);
PyObject* rqs_make_options_dict(const struct rqs_control_type *control);
PyObject* rqs_make_inform_dict(const struct rqs_inform_type *inform);

/* Module global variables */
static void *data;                       // private internal data
static struct nrek_control_type control;  // control struct
static struct nrek_inform_type inform;    // inform struct
static bool init_called = false;         // record if initialise was called
static bool load_called = false;         // record if load was called
static bool load_s_called = false;       // record if load_s was called
static int status = 0;                   // exit status

//  *-*-*-*-*-*-*-*-*-*-   UPDATE CONTROL    -*-*-*-*-*-*-*-*-*-*

/* Update the control options: use C defaults but update any passed via Python*/
// NB not static as it is used for nested control within TRU Python interface
bool nrek_update_control(struct nrek_control_type *control,
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
        if(strcmp(key_name, "eks_max") == 0){
            if(!parse_int_option(value, "eks_max",
                                  &control->eks_max))
                return false;
            continue;
        }
        if(strcmp(key_name, "it_max") == 0){
            if(!parse_int_option(value, "it_max",
                                  &control->it_max))
                return false;
            continue;
        }
        if(strcmp(key_name, "f") == 0){
            if(!parse_double_option(value, "f",
                                  &control->f))
                return false;
            continue;
        }
        if(strcmp(key_name, "increase") == 0){
            if(!parse_double_option(value, "increase",
                                  &control->increase))
                return false;
            continue;
        }
        if(strcmp(key_name, "stop_residual") == 0){
            if(!parse_double_option(value, "stop_residual",
                                  &control->stop_residual))
                return false;
            continue;
        }
        if(strcmp(key_name, "reorthogonalize") == 0){
            if(!parse_bool_option(value, "reorthogonalize",
                                  &control->reorthogonalize))
                return false;
            continue;
        }
        if(strcmp(key_name, "s_version_52") == 0){
            if(!parse_bool_option(value, "s_version_52",
                                  &control->s_version_52))
                return false;
            continue;
        }
        if(strcmp(key_name, "perturb_c") == 0){
            if(!parse_bool_option(value, "perturb_c",
                                  &control->perturb_c))
                return false;
            continue;
        }
        if(strcmp(key_name, "stop_check_all_orders") == 0){
            if(!parse_bool_option(value, "stop_check_all_orders",
                                  &control->stop_check_all_orders))
                return false;
            continue;
        }
        if(strcmp(key_name, "new_weight") == 0){
            if(!parse_bool_option(value, "new_weight",
                                  &control->new_weight))
                return false;
            continue;
        }
        if(strcmp(key_name, "new_values") == 0){
            if(!parse_bool_option(value, "new_values",
                                  &control->new_values))
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
        if(strcmp(key_name, "linear_solver") == 0){
            if(!parse_char_option(value, "linear_solver",
                                  control->linear_solver,
                                  sizeof(control->linear_solver)))
                return false;
            continue;
        }
        if(strcmp(key_name, "linear_solver_for_s") == 0){
            if(!parse_char_option(value, "linear_solver_for_s",
                                  control->linear_solver_for_s,
                                  sizeof(control->linear_solver_for_s)))
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
        if(strcmp(key_name, "sls_s_options") == 0){
            if(!sls_update_control(&control->sls_s_control, value))
                return false;
            continue;
        }
        if(strcmp(key_name, "rqs_options") == 0){
            if(!rqs_update_control(&control->rqs_control, value))
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
// NB not static as it is used for nested inform within TRU Python interface
PyObject* nrek_make_options_dict(const struct nrek_control_type *control){
    PyObject *py_options = PyDict_New();

    PyDict_SetItemString(py_options, "error",
                         PyLong_FromLong(control->error));
    PyDict_SetItemString(py_options, "out",
                         PyLong_FromLong(control->out));
    PyDict_SetItemString(py_options, "print_level",
                         PyLong_FromLong(control->print_level));
    PyDict_SetItemString(py_options, "eks_max",
                         PyLong_FromLong(control->eks_max));
    PyDict_SetItemString(py_options, "it_max",
                         PyLong_FromLong(control->it_max));
    PyDict_SetItemString(py_options, "f",
                         PyFloat_FromDouble(control->f));
    PyDict_SetItemString(py_options, "increase",
                         PyFloat_FromDouble(control->increase));
    PyDict_SetItemString(py_options, "stop_residual",
                         PyFloat_FromDouble(control->stop_residual));
    PyDict_SetItemString(py_options, "reorthogonalize",
                         PyBool_FromLong(control->reorthogonalize));
    PyDict_SetItemString(py_options, "s_version_52",
                         PyBool_FromLong(control->s_version_52));
    PyDict_SetItemString(py_options, "perturb_c",
                         PyBool_FromLong(control->perturb_c));
    PyDict_SetItemString(py_options, "stop_check_all_orders",
                         PyBool_FromLong(control->stop_check_all_orders));
    PyDict_SetItemString(py_options, "new_weight",
                         PyBool_FromLong(control->new_weight));
    PyDict_SetItemString(py_options, "new_values",
                         PyBool_FromLong(control->new_values));
    PyDict_SetItemString(py_options, "space_critical",
                         PyBool_FromLong(control->space_critical));
    PyDict_SetItemString(py_options, "deallocate_error_fatal",
                         PyBool_FromLong(control->deallocate_error_fatal));
    PyDict_SetItemString(py_options, "linear_solver",
                         PyUnicode_FromString(control->linear_solver));
    PyDict_SetItemString(py_options, "linear_solver_for_s",
                         PyUnicode_FromString(control->linear_solver_for_s));
    PyDict_SetItemString(py_options, "prefix",
                         PyUnicode_FromString(control->prefix));
    PyDict_SetItemString(py_options, "sls_options",
                         sls_make_options_dict(&control->sls_control));
    PyDict_SetItemString(py_options, "sls_s_options",
                         sls_make_options_dict(&control->sls_s_control));
    PyDict_SetItemString(py_options, "rqs_options",
                         rqs_make_options_dict(&control->rqs_control));

    return py_options;
}

//  *-*-*-*-*-*-*-*-*-*-   MAKE TIME    -*-*-*-*-*-*-*-*-*-*

/* Take the time struct from C and turn it into a python dictionary */
static PyObject* nrek_make_time_dict(const struct nrek_time_type *time){
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

//  *-*-*-*-*-*-*-*-*-*-   MAKE INFORM    -*-*-*-*-*-*-*-*-*-*

/* Take the inform struct from C and turn it into a python dictionary */
// NB not static as it is used for nested control within TRU Python interface
PyObject* nrek_make_inform_dict(const struct nrek_inform_type *inform){
    PyObject *py_inform = PyDict_New();

    PyDict_SetItemString(py_inform, "status",
                         PyLong_FromLong(inform->status));
    PyDict_SetItemString(py_inform, "alloc_status",
                         PyLong_FromLong(inform->alloc_status));
    PyDict_SetItemString(py_inform, "iter",
                         PyLong_FromLong(inform->iter));
    PyDict_SetItemString(py_inform, "n_vec",
                         PyLong_FromLong(inform->n_vec));
    PyDict_SetItemString(py_inform, "obj",
                         PyFloat_FromDouble(inform->obj));
    PyDict_SetItemString(py_inform, "obj_regularized",
                         PyFloat_FromDouble(inform->obj_regularized));
    PyDict_SetItemString(py_inform, "x_norm",
                         PyFloat_FromDouble(inform->x_norm));
    PyDict_SetItemString(py_inform, "multiplier",
                         PyFloat_FromDouble(inform->multiplier));
    PyDict_SetItemString(py_inform, "weight",
                         PyFloat_FromDouble(inform->weight));
    PyDict_SetItemString(py_inform, "next_weight",
                         PyFloat_FromDouble(inform->next_weight));
    PyDict_SetItemString(py_inform, "error",
                         PyFloat_FromDouble(inform->error));
    PyDict_SetItemString(py_inform, "bad_alloc",
                         PyUnicode_FromString(inform->bad_alloc));

    // Set time nested dictionary
    PyDict_SetItemString(py_inform, "time",
                         nrek_make_time_dict(&inform->time));

    // Set dictionaries from subservient packages
    PyDict_SetItemString(py_inform, "sls_inform",
                         sls_make_inform_dict(&inform->sls_inform));
    PyDict_SetItemString(py_inform, "sls_s_inform",
                         sls_make_inform_dict(&inform->sls_s_inform));
    PyDict_SetItemString(py_inform, "rqs_inform",
                         rqs_make_inform_dict(&inform->rqs_inform));

    return py_inform;
}

//  *-*-*-*-*-*-*-*-*-*-   NREK_INITIALIZE    -*-*-*-*-*-*-*-*-*-*

static PyObject* py_nrek_initialize(PyObject *self){

    // Call nrek_initialize
    nrek_initialize(&data, &control, &status);

    // Record that nrek has been initialised
    init_called = true;

    // Return options Python dictionary
    PyObject *py_options = nrek_make_options_dict(&control);
    return Py_BuildValue("O", py_options);
}

//  *-*-*-*-*-*-*-*-*-*-*-*-   NREK_LOAD    -*-*-*-*-*-*-*-*-*-*-*-*

static PyObject* py_nrek_load(PyObject *self, PyObject *args, PyObject *keywds){
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
    nrek_reset_control(&control, &data, &status);

    // Update nrek control options
    if(!nrek_update_control(&control, py_options))
        return NULL;

    // Call nrek_import
    nrek_import(&control, &data, &status, n,
               H_type, H_ne, H_row, H_col, H_ptr);

    // Free allocated memory
    if(H_row != NULL) free(H_row);
    if(H_col != NULL) free(H_col);
    if(H_ptr != NULL) free(H_ptr);

    // Raise any status errors
    if(!check_error_codes(status))
        return NULL;

    // Record that nrek structures been initialised
    load_called = true;

    // Return None boilerplate
    Py_INCREF(Py_None);
    return Py_None;
}

//  *-*-*-*-*-*-*-*-*-*-*-*-   NREK_LOAD_S    -*-*-*-*-*-*-*-*-*-*-*-*

static PyObject* py_nrek_load_s(PyObject *self, PyObject *args,
                               PyObject *keywds){
    PyArrayObject *py_S_row, *py_S_col, *py_S_ptr;
    PyObject *py_options = NULL;
    int *S_row = NULL, *S_col = NULL, *S_ptr = NULL;
    const char *S_type;
    int n, S_ne;

    // Check that package has been initialised
    if(!check_init(init_called))
        return NULL;

    // Parse positional and keyword arguments
    static char *kwlist[] = {"n","S_type","S_ne","S_row","S_col","S_ptr",
                             "options",NULL};

    if(!PyArg_ParseTupleAndKeywords(args, keywds, "isiOOO|O", kwlist, &n,
                                    &S_type, &S_ne, &py_S_row,
                                    &py_S_col, &py_S_ptr,
                                    &py_options))
        return NULL;

    // Check that array inputs are of correct type, size, and shape

    if(!(
        check_array_int("S_row", py_S_row, S_ne) &&
        check_array_int("S_col", py_S_col, S_ne) &&
        check_array_int("S_ptr", py_S_ptr, n+1)
        ))
        return NULL;

    // Convert 64bit integer S_row array to 32bit
    if((PyObject *) py_S_row != Py_None){
        S_row = malloc(S_ne * sizeof(int));
        long int *S_row_long = (long int *) PyArray_DATA(py_S_row);
        for(int i = 0; i < S_ne; i++) S_row[i] = (int) S_row_long[i];
    }

    // Convert 64bit integer S_col array to 32bit
    if((PyObject *) py_S_col != Py_None){
        S_col = malloc(S_ne * sizeof(int));
        long int *S_col_long = (long int *) PyArray_DATA(py_S_col);
        for(int i = 0; i < S_ne; i++) S_col[i] = (int) S_col_long[i];
    }

    // Convert 64bit integer S_ptr array to 32bit
    if((PyObject *) py_S_ptr != Py_None){
        S_ptr = malloc((n+1) * sizeof(int));
        long int *S_ptr_long = (long int *) PyArray_DATA(py_S_ptr);
        for(int i = 0; i < n+1; i++) S_ptr[i] = (int) S_ptr_long[i];
    }

    // Reset control options
    nrek_reset_control(&control, &data, &status);

    // Update nrek control options
    if(!nrek_update_control(&control, py_options))
        return NULL;

    // Call nrek_import
    nrek_s_import(&data, &status, n, S_type, S_ne, S_row, S_col, S_ptr);

    // Free allocated memory
    if(S_row != NULL) free(S_row);
    if(S_col != NULL) free(S_col);
    if(S_ptr != NULL) free(S_ptr);

    // Raise any status errors
    if(!check_error_codes(status))
        return NULL;

    // Record that nrek S structure been initialised
    load_s_called = true;

    // Return None boilerplate
    Py_INCREF(Py_None);
    return Py_None;
}

//  *-*-*-*-*-*-*-*-*-*-*-*-   NREK_RESET_OPTIONS    -*-*-*-*-*-*-*-*-*-*-*-*

static PyObject* py_nrek_reset_options(PyObject *self, PyObject *args, 
                                      PyObject *keywds){
    PyObject *py_options = NULL;

    // Check that package has been initialised
    if(!check_init(init_called))
        return NULL;

    // Parse positional and keyword arguments
    static char *kwlist[] = {"options",NULL};

    if(!PyArg_ParseTupleAndKeywords(args, keywds, "O", kwlist, &py_options))
        return NULL;

    // Reset control options
    nrek_reset_control(&control, &data, &status);

    // Update nrek control options
    if(!nrek_update_control(&control, py_options))
        return NULL;

    // Return None boilerplate
    Py_INCREF(Py_None);
    return Py_None;
}


//  *-*-*-*-*-*-*-*-*-*-   NREK_SOLVE_PROBLEM   -*-*-*-*-*-*-*-*

static PyObject* py_nrek_solve_problem(PyObject *self, PyObject *args,
                                       PyObject *keywds){
    PyArrayObject *py_g, *py_H_val, *py_S_val = NULL;
    double *g, *H_val, *S_val = NULL;
    int n, H_ne, S_ne;
    double power, weight;

    // Check that package has been initialised
    if(!check_load(load_called))
        return NULL;

    // Parse positional and keyword arguments
    static char *kwlist[] = {"n","H_ne","H_val","g","power","weight",
                             "S_ne","S_val",NULL};

    if(!PyArg_ParseTupleAndKeywords(args, keywds, "iiOOdd|iO", kwlist,
                                    &n, &H_ne, &py_H_val, &py_g, 
                                    &power, &weight, &S_ne, &py_S_val))
        return NULL;

    // Check that array inputs are of correct type, size, and shape
    if(!check_array_double("g", py_g, n))
        return NULL;
    if(!check_array_double("H_val", py_H_val, H_ne))
        return NULL;
    if(load_s_called) {
      if(!check_array_double("S_val", py_S_val, S_ne))
          return NULL;
    }

    // Get array data pointer
    g = (double *) PyArray_DATA(py_g);
    H_val = (double *) PyArray_DATA(py_H_val);
    if(py_S_val != NULL) S_val = (double *) PyArray_DATA(py_S_val);

   // Create NumPy output arrays
    npy_intp ndim[] = {n}; // size of x
    PyArrayObject *py_x =
      (PyArrayObject *) PyArray_SimpleNew(1, ndim, NPY_DOUBLE);
    double *x = (double *) PyArray_DATA(py_x);

    // Call nrek_solve_problem
    nrek_solve_problem(&data, &status, n, H_ne, H_val, g, power, weight, x,
                       S_ne, S_val);
    // for( int i = 0; i < n; i++) printf("x %f\n", x[i]);

    // Propagate any errors with the callback function
    if(PyErr_Occurred())
        return NULL;

    // Raise any status errors
    if(!check_error_codes(status))
        return NULL;

    // Return x
    PyObject *solve_problem_return;

    solve_problem_return = Py_BuildValue("O", py_x);
    Py_INCREF(solve_problem_return);
    return solve_problem_return;

}

//  *-*-*-*-*-*-*-*-*-*-   NREK_INFORMATION   -*-*-*-*-*-*-*-*

static PyObject* py_nrek_information(PyObject *self){

    // Check that package has been initialised
    if(!check_init(init_called))
        return NULL;

    // Call nrek_information
    nrek_information(&data, &inform, &status);

    // Return status and inform Python dictionary
    PyObject *py_inform = nrek_make_inform_dict(&inform);
    return Py_BuildValue("O", py_inform);
}

//  *-*-*-*-*-*-*-*-*-*-   NREK_TERMINATE   -*-*-*-*-*-*-*-*-*-*

static PyObject* py_nrek_terminate(PyObject *self){

    // Check that package has been initialised
    if(!check_init(init_called))
        return NULL;

    // Call nrek_terminate
    nrek_terminate(&data, &control, &inform);

    // require future calls start with nrek_initialize
    init_called = false;
    load_called = false;
    load_s_called = false;

    // Return None boilerplate
    Py_INCREF(Py_None);
    return Py_None;
}

//  *-*-*-*-*-*-*-*-*-*-   INITIALIZE NREK PYTHON MODULE    -*-*-*-*-*-*-*-*-*-*

/* nrek python module method table */
static PyMethodDef nrek_module_methods[] = {
    {"initialize", (PyCFunction) py_nrek_initialize, METH_NOARGS, NULL},
    {"load", (PyCFunction) py_nrek_load, METH_VARARGS | METH_KEYWORDS, NULL},
    {"load_s", (PyCFunction) py_nrek_load_s, METH_VARARGS | METH_KEYWORDS, NULL},
    {"reset_options", (PyCFunction) py_nrek_reset_options, METH_VARARGS | METH_KEYWORDS, NULL},
    {"solve_problem", (PyCFunction) py_nrek_solve_problem, METH_VARARGS | METH_KEYWORDS, NULL},
    {"information", (PyCFunction) py_nrek_information, METH_NOARGS, NULL},
    {"terminate", (PyCFunction) py_nrek_terminate, METH_NOARGS, NULL},
    {NULL, NULL, 0, NULL}  /* Sentinel */
};

/* nrek python module documentation */

PyDoc_STRVAR(nrek_module_doc,

"The nrek package uses an extended-Krylov-subspace iteration to find the\n"
"global minimizer of a norm-regularized quadratic objective function;\n"
" this is commonly known as the regularization subproblem.\n"
"The aim is to minimize the regularized quadratic objective function\n"
"r(x) = f + g^T x + 1/2 x^T H x + sigma/p ||x||_S^p, \n"
"where the S-norm of x is defined to be ||x||_S = sqrt{x^T S x},\n"
"and where the power p > 2 and the weight sigma > 0.\n"
"The matrix S need not be provided in the commonly-occurring\n"
"l_2-regularization case for which S = I, the n by n identity matrix.\n"
"\n"
"A single factorization of matrices H and (when present) S\n"
"will be required, so this package is most suited\n"
"for the case where such a factorization may be found efficiently. If\n"
"this is not the case, the package gltr may be preferred.\n"
"\n"
"See $GALAHAD/html/Python/nrek.html for argument lists, call order\n"
"and other details.\n"
"\n"
);

/* nrek python module definition */
static struct PyModuleDef module = {
   PyModuleDef_HEAD_INIT,
   "nrek",               /* name of module */
   nrek_module_doc,      /* module documentation, may be NULL */
   -1,                  /* size of per-interpreter state of the module,or -1
                           if the module keeps state in global variables */
   nrek_module_methods   /* module methods */
};

/* Python module initialization */
PyMODINIT_FUNC PyInit_nrek(void) { // must be same as module name above
    import_array();  // for NumPy arrays
    return PyModule_Create(&module);
}
