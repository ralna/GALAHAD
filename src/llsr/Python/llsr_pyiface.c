//* \file llsr_pyiface.c */

/*
 * THIS VERSION: GALAHAD 4.1 - 2023-06-22 AT 07:30 GMT.
 *
 *-*-*-*-*-*-*-*-*-  GALAHAD_LLSR PYTHON INTERFACE  *-*-*-*-*-*-*-*-*-*-
 *
 *  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
 *  Principal author: Jaroslav Fowkes & Nick Gould
 *
 *  History -
 *   originally released GALAHAD Version 4.1. June 5th 2023
 *
 *  For full documentation, see
 *   http://galahad.rl.ac.uk/galahad-www/specs.html
 */

#include "galahad_python.h"
#include "galahad_llsr.h"

/* Nested SBLS, SLS and IR control and inform prototypes */
bool sbls_update_control(struct sbls_control_type *control,
                         PyObject *py_options);
PyObject* sbls_make_options_dict(const struct sbls_control_type *control);
PyObject* sbls_make_inform_dict(const struct sbls_inform_type *inform);
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
static struct llsr_control_type control;  // control struct
static struct llsr_inform_type inform;    // inform struct
static bool init_called = false;         // record if initialise was called
static bool load_called = false;         // record if load was called
static bool load_scaling_called = false; // record if load_scaling was called
static int status = 0;                   // exit status

//  *-*-*-*-*-*-*-*-*-*-   UPDATE CONTROL    -*-*-*-*-*-*-*-*-*-*

/* Update the control options: use C defaults but update any passed via Python*/
// NB not static as it is used for nested control within TRU Python interface
bool llsr_update_control(struct llsr_control_type *control,
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
        if(strcmp(key_name, "new_a") == 0){
            if(!parse_int_option(value, "new_a",
                                  &control->new_a))
                return false;
            continue;
        }
        if(strcmp(key_name, "new_s") == 0){
            if(!parse_int_option(value, "new_s",
                                  &control->new_s))
                return false;
            continue;
        }
        if(strcmp(key_name, "max_factorizations") == 0){
            if(!parse_int_option(value, "max_factorizations",
                                  &control->max_factorizations))
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
        if(strcmp(key_name, "use_initial_multiplier") == 0){
            if(!parse_bool_option(value, "use_initial_multiplier",
                                  &control->use_initial_multiplier))
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
        if(strcmp(key_name, "sbls_options") == 0){
            if(!sbls_update_control(&control->sbls_control, value))
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
// NB not static as it is used for nested inform within TRU Python interface
PyObject* llsr_make_options_dict(const struct llsr_control_type *control){
    PyObject *py_options = PyDict_New();
    PyDict_SetItemString(py_options, "error",
                         PyLong_FromLong(control->error));
    PyDict_SetItemString(py_options, "out",
                         PyLong_FromLong(control->out));
    PyDict_SetItemString(py_options, "print_level",
                         PyLong_FromLong(control->print_level));
    PyDict_SetItemString(py_options, "new_a",
                         PyLong_FromLong(control->new_a));
    PyDict_SetItemString(py_options, "new_s",
                         PyLong_FromLong(control->new_s));
    PyDict_SetItemString(py_options, "max_factorizations",
                         PyLong_FromLong(control->max_factorizations));
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
    PyDict_SetItemString(py_options, "use_initial_multiplier",
                         PyBool_FromLong(control->use_initial_multiplier));
    PyDict_SetItemString(py_options, "space_critical",
                         PyBool_FromLong(control->space_critical));
    PyDict_SetItemString(py_options, "deallocate_error_fatal",
                         PyBool_FromLong(control->deallocate_error_fatal));
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
static PyObject* llsr_make_time_dict(const struct llsr_time_type *time){
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
static PyObject* llsr_make_history_dict(const struct llsr_history_type *history){
    PyObject *py_history = PyDict_New();

    // Set float/double history entries
    PyDict_SetItemString(py_history, "lambda",
                         PyFloat_FromDouble(history->lambda));
    PyDict_SetItemString(py_history, "x_norm",
                         PyFloat_FromDouble(history->x_norm));
    PyDict_SetItemString(py_history, "r_norm",
                         PyFloat_FromDouble(history->r_norm));

    return py_history;
}

//  *-*-*-*-*-*-*-*-*-*-   MAKE INFORM    -*-*-*-*-*-*-*-*-*-*

/* Take the inform struct from C and turn it into a python dictionary */
// NB not static as it is used for nested control within TRU Python interface
PyObject* llsr_make_inform_dict(const struct llsr_inform_type *inform){
    PyObject *py_inform = PyDict_New();

    PyDict_SetItemString(py_inform, "status",
                         PyLong_FromLong(inform->status));
    PyDict_SetItemString(py_inform, "alloc_status",
                         PyLong_FromLong(inform->alloc_status));
    PyDict_SetItemString(py_inform, "factorizations",
                         PyLong_FromLong(inform->factorizations));
    PyDict_SetItemString(py_inform, "len_history",
                         PyLong_FromLong(inform->len_history));
    PyDict_SetItemString(py_inform, "r_norm",
                         PyFloat_FromDouble(inform->r_norm));
    PyDict_SetItemString(py_inform, "x_norm",
                         PyFloat_FromDouble(inform->x_norm));
    PyDict_SetItemString(py_inform, "multiplier",
                         PyFloat_FromDouble(inform->multiplier));
    PyDict_SetItemString(py_inform, "bad_alloc",
                         PyUnicode_FromString(inform->bad_alloc));

    // include history arrays
    //npy_intp hdim[] = {100};
    //PyArrayObject *py_lambda =
    //  (PyArrayObject*) PyArray_SimpleNew(1, hdim, NPY_DOUBLE);
    //double *lambda = (double *) PyArray_DATA(py_lambda);
    //for(int i=0; i<100; i++) lambda[i] = inform->history[i]->lambda;
    //PyDict_SetItemString(py_inform, "lambda", (PyObject *) py_lambda);
    //PyArrayObject *py_x_norm =
    //  (PyArrayObject*) PyArray_SimpleNew(1, hdim, NPY_DOUBLE);
    //double *x_norm = (double *) PyArray_DATA(py_x_norm);
    //for(int i=0; i<100; i++) x_norm[i] = inform->history[i]->x_norm;
    //PyDict_SetItemString(py_inform, "x_norm", (PyObject *) py_x_norm);

    // Set time nested dictionary
    PyDict_SetItemString(py_inform, "time",
                         llsr_make_time_dict(&inform->time));

    // Set dictionaries from subservient packages
    PyDict_SetItemString(py_inform, "sbls_inform",
                         sbls_make_inform_dict(&inform->sbls_inform));
    PyDict_SetItemString(py_inform, "sls_inform",
                         sls_make_inform_dict(&inform->sls_inform));
    PyDict_SetItemString(py_inform, "ir_inform",
                         ir_make_inform_dict(&inform->ir_inform));

    return py_inform;
}

//  *-*-*-*-*-*-*-*-*-*-   LLSR_INITIALIZE    -*-*-*-*-*-*-*-*-*-*

static PyObject* py_llsr_initialize(PyObject *self){

    // Call llsr_initialize
    llsr_initialize(&data, &control, &status);

    // Record that llsr has been initialised
    init_called = true;

    // Return options Python dictionary
    PyObject *py_options = llsr_make_options_dict(&control);
    return Py_BuildValue("O", py_options);
}

//  *-*-*-*-*-*-*-*-*-*-*-*-   LLSR_LOAD    -*-*-*-*-*-*-*-*-*-*-*-*

static PyObject* py_llsr_load(PyObject *self, PyObject *args, PyObject *keywds){
    PyArrayObject *py_A_row, *py_A_col, *py_A_ptr;
    PyObject *py_options = NULL;
    int *A_row = NULL, *A_col = NULL, *A_ptr = NULL;
    const char *A_type;
    int m, n, A_ne;

    // Check that package has been initialised
    if(!check_init(init_called))
        return NULL;

    // Parse positional and keyword arguments
    static char *kwlist[] = {"m","n","A_type","A_ne","A_row","A_col","A_ptr",
                             "options",NULL};

    if(!PyArg_ParseTupleAndKeywords(args, keywds, "iisiOOO|O", kwlist, &m, &n,
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
    llsr_reset_control(&control, &data, &status);

    // Update llsr control options
    if(!llsr_update_control(&control, py_options))
        return NULL;

    // Call llsr_import
    llsr_import(&control, &data, &status, m, n,
                A_type, A_ne, A_row, A_col, A_ptr);

    // Free allocated memory
    if(A_row != NULL) free(A_row);
    if(A_col != NULL) free(A_col);
    if(A_ptr != NULL) free(A_ptr);

    // Raise any status errors
    if(!check_error_codes(status))
        return NULL;

    // Record that llsr structures been initialised
    load_called = true;

    // Return None boilerplate
    Py_INCREF(Py_None);
    return Py_None;
}

//  *-*-*-*-*-*-*-*-*-*-*-*-   LLSR_LOAD_SCALING    -*-*-*-*-*-*-*-*-*-*-*-*

static PyObject* py_llsr_load_scaling(PyObject *self, PyObject *args,
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
    llsr_reset_control(&control, &data, &status);

    // Update llsr control options
    if(!llsr_update_control(&control, py_options))
        return NULL;

    // Call llsr_import
    llsr_import_scaling(&control, &data, &status, n,
                        S_type, S_ne, S_row, S_col, S_ptr);

    // Free allocated memory
    if(S_row != NULL) free(S_row);
    if(S_col != NULL) free(S_col);
    if(S_ptr != NULL) free(S_ptr);

    // Raise any status errors
    if(!check_error_codes(status))
        return NULL;

    // Record that llsr M structure been initialised
    load_scaling_called = true;

    // Return None boilerplate
    Py_INCREF(Py_None);
    return Py_None;
}

//  *-*-*-*-*-*-*-*-*-*-   LLSR_SOLVE_PROBLEM   -*-*-*-*-*-*-*-*

static PyObject* py_llsr_solve_problem(PyObject *self, PyObject *args,
                                       PyObject *keywds){
    PyArrayObject *py_A_val, *py_b, *py_S_val = NULL;
    double *A_val, *b, *S_val = NULL;
    int m, n, A_ne, S_ne;
    double power, weight;

    // Check that package has been initialised
    if(!check_load(load_called))
        return NULL;

    // Parse positional and keyword arguments
    static char *kwlist[] = {"m","n","power", "weight","A_ne","A_val","b",
                             "S_ne","S_val",NULL};

    if(!PyArg_ParseTupleAndKeywords(args, keywds, "iiddiOO|iO", kwlist,
                                    &m, &n, &power, &weight, &A_ne, &py_A_val,
                                    &py_b, &S_ne, &py_S_val))
        return NULL;

    // Check that array inputs are of correct type, size, and shape
    if(!check_array_double("b", py_b, m))
        return NULL;
    if(!check_array_double("A_val", py_A_val, A_ne))
        return NULL;
    if(load_scaling_called) {
      if(!check_array_double("S_val", py_S_val, S_ne))
          return NULL;
    }

    // Get array data pointer
    b = (double *) PyArray_DATA(py_b);
    A_val = (double *) PyArray_DATA(py_A_val);
    if(py_S_val != NULL) S_val = (double *) PyArray_DATA(py_S_val);

   // Create NumPy output arrays
    npy_intp ndim[] = {n}; // size of x
    PyArrayObject *py_x =
      (PyArrayObject *) PyArray_SimpleNew(1, ndim, NPY_DOUBLE);
    double *x = (double *) PyArray_DATA(py_x);

    // Call llsr_solve_problem
    llsr_solve_problem(&data, &status, m, n, power, weight, A_ne, A_val, b, x,
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

//  *-*-*-*-*-*-*-*-*-*-   LLSR_INFORMATION   -*-*-*-*-*-*-*-*

static PyObject* py_llsr_information(PyObject *self){

    // Check that package has been initialised
    if(!check_init(init_called))
        return NULL;

    // Call llsr_information
    llsr_information(&data, &inform, &status);

    // Return status and inform Python dictionary
    PyObject *py_inform = llsr_make_inform_dict(&inform);
    return Py_BuildValue("O", py_inform);
}

//  *-*-*-*-*-*-*-*-*-*-   LLSR_TERMINATE   -*-*-*-*-*-*-*-*-*-*

static PyObject* py_llsr_terminate(PyObject *self){

    // Check that package has been initialised
    if(!check_init(init_called))
        return NULL;

    // Call llsr_terminate
    llsr_terminate(&data, &control, &inform);

    // require future calls start with llsr_initialize
    init_called = false;
    load_called = false;
    load_scaling_called = false;

    // Return None boilerplate
    Py_INCREF(Py_None);
    return Py_None;
}

//  *-*-*-*-*-*-*-*-*-*-   INITIALIZE LLSR PYTHON MODULE    -*-*-*-*-*-*-*-*-*-*

/* llsr python module method table */
static PyMethodDef llsr_module_methods[] = {
    {"initialize", (PyCFunction) py_llsr_initialize, METH_NOARGS, NULL},
    {"load", (PyCFunction) py_llsr_load, METH_VARARGS | METH_KEYWORDS, NULL},
    {"load_scaling", (PyCFunction) py_llsr_load_scaling, METH_VARARGS | METH_KEYWORDS, NULL},
    {"solve_problem", (PyCFunction) py_llsr_solve_problem, METH_VARARGS | METH_KEYWORDS, NULL},
    {"information", (PyCFunction) py_llsr_information, METH_NOARGS, NULL},
    {"terminate", (PyCFunction) py_llsr_terminate, METH_NOARGS, NULL},
    {NULL, NULL, 0, NULL}  /* Sentinel */
};

/* llsr python module documentation */

PyDoc_STRVAR(llsr_module_doc,

"Given a real m by n matrix A, a real n by n symmetric diagonally-dominant\n"
"matrix S, a real m vector b and scalars sigma>0 and $p>=2$,\n"
"the llsr package finds a minimizer of the regularized\n"
"linear least-squares objective function\n"
"   1/2 || A x  - b ||_2^w + sigma/p ||x||_S^p,\n"
"where the S-norm of x is  ||x||_S = sqrt{x^T S x}.\n"
"This problem commonly occurs as a subproblem in nonlinear\n"
"least-squares calculations.\n"
"The matrix S need not be provided in the commonly-occurring\n"
"l_2-regularization case for which S = I, the n by n identity matrix.\n"
"\n"
"Factorization of matrices of the form A^T A + lambda S, or\n"
"( lambda S  A^T )\n"
"(    A      -I  )\n"
"for a succession of scalars lambda will be required, so this package is\n"
"most suited for the case where such a factorization may be found\n"
"efficiently. If this is not the case, the package gltr may be preferred.\n"
"\n"
"See $GALAHAD/html/Python/llsr.html for argument lists, call order\n"
"and other details.\n"
"\n"
);

/* llsr python module definition */
static struct PyModuleDef module = {
   PyModuleDef_HEAD_INIT,
   "llsr",               /* name of module */
   llsr_module_doc,      /* module documentation, may be NULL */
   -1,                  /* size of per-interpreter state of the module,or -1
                           if the module keeps state in global variables */
   llsr_module_methods   /* module methods */
};

/* Python module initialization */
PyMODINIT_FUNC PyInit_llsr(void) { // must be same as module name above
    import_array();  // for NumPy arrays
    return PyModule_Create(&module);
}
