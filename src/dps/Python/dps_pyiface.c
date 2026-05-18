//* \file dps_pyiface.c */

/*
 * THIS VERSION: GALAHAD 5.0 - 2024-06-16 AT 10:40 GMT.
 *
 *-*-*-*-*-*-*-*-*-  GALAHAD_DPS PYTHON INTERFACE  *-*-*-*-*-*-*-*-*-*-
 *
 *  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
 *  Principal author: Jaroslav Fowkes & Nick Gould
 *
 *  History -
 *   originally released GALAHAD Version 4.1. March 13th 2023
 *
 *  For full documentation, see
 *   http://galahad.rl.ac.uk/galahad-www/specs.html
 */

#include "galahad_python.h"
#include "galahad_dps.h"

/* Nested SLS control and inform prototypes */
bool sls_update_control(struct sls_control_type *control,
                        PyObject *py_options);
PyObject* sls_make_options_dict(const struct sls_control_type *control);
PyObject* sls_make_inform_dict(const struct sls_inform_type *inform);

/* Module global variables */
static void *data;                       // private internal data
static struct dps_control_type control;  // control struct
static struct dps_inform_type inform;    // inform struct
static bool init_called = false;         // record if initialise was called
static bool load_called = false;         // record if load was called
static int status = 0;                   // exit status

//  *-*-*-*-*-*-*-*-*-*-   UPDATE CONTROL    -*-*-*-*-*-*-*-*-*-*

/* Update the control options: use C defaults but update any passed via Python*/
// NB not static as it is used for nested control within TRU Python interface
bool dps_update_control(struct dps_control_type *control,
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
        if(strcmp(key_name, "new_h") == 0){
            if(!parse_int_option(value, "new_h",
                                  &control->new_h))
                return false;
            continue;
        }
        if(strcmp(key_name, "taylor_max_degree") == 0){
            if(!parse_int_option(value, "taylor_max_degree",
                                  &control->taylor_max_degree))
                return false;
            continue;
        }
        if(strcmp(key_name, "eigen_min") == 0){
            if(!parse_double_option(value, "eigen_min",
                                  &control->eigen_min))
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
        if(strcmp(key_name, "stop_absolute_normal") == 0){
            if(!parse_double_option(value, "stop_absolute_normal",
                                  &control->stop_absolute_normal))
                return false;
            continue;
        }
        if(strcmp(key_name, "goldfarb") == 0){
            if(!parse_bool_option(value, "goldfarb",
                                  &control->goldfarb))
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
PyObject* dps_make_options_dict(const struct dps_control_type *control){
    PyObject *py_options = PyDict_New();

    PyDict_SetItemString(py_options, "error",
                         PyLong_FromLong(control->error));
    PyDict_SetItemString(py_options, "out",
                         PyLong_FromLong(control->out));
    PyDict_SetItemString(py_options, "problem",
                         PyLong_FromLong(control->problem));
    PyDict_SetItemString(py_options, "print_level",
                         PyLong_FromLong(control->print_level));
    PyDict_SetItemString(py_options, "new_h",
                         PyLong_FromLong(control->new_h));
    PyDict_SetItemString(py_options, "taylor_max_degree",
                         PyLong_FromLong(control->taylor_max_degree));
    PyDict_SetItemString(py_options, "eigen_min",
                         PyFloat_FromDouble(control->eigen_min));
    PyDict_SetItemString(py_options, "lower",
                         PyFloat_FromDouble(control->lower));
    PyDict_SetItemString(py_options, "upper",
                         PyFloat_FromDouble(control->upper));
    PyDict_SetItemString(py_options, "stop_normal",
                         PyFloat_FromDouble(control->stop_normal));
    PyDict_SetItemString(py_options, "stop_absolute_normal",
                         PyFloat_FromDouble(control->stop_absolute_normal));
    PyDict_SetItemString(py_options, "goldfarb",
                         PyBool_FromLong(control->goldfarb));
    PyDict_SetItemString(py_options, "space_critical",
                         PyBool_FromLong(control->space_critical));
    PyDict_SetItemString(py_options, "deallocate_error_fatal",
                         PyBool_FromLong(control->deallocate_error_fatal));
    PyDict_SetItemString(py_options, "problem_file",
                         PyUnicode_FromString(control->problem_file));
    PyDict_SetItemString(py_options, "symmetric_linear_solver",
                         PyUnicode_FromString(control->symmetric_linear_solver));
    PyDict_SetItemString(py_options, "prefix",
                         PyUnicode_FromString(control->prefix));
    PyDict_SetItemString(py_options, "sls_options",
                         sls_make_options_dict(&control->sls_control));

    return py_options;
}

//  *-*-*-*-*-*-*-*-*-*-   MAKE TIME    -*-*-*-*-*-*-*-*-*-*

/* Take the time struct from C and turn it into a python dictionary */
static PyObject* dps_make_time_dict(const struct dps_time_type *time){
    PyObject *py_time = PyDict_New();

    // Set float/double time entries

    PyDict_SetItemString(py_time, "total",
                         PyFloat_FromDouble(time->total));
    PyDict_SetItemString(py_time, "analyse",
                         PyFloat_FromDouble(time->analyse));
    PyDict_SetItemString(py_time, "factorize",
                         PyFloat_FromDouble(time->factorize));
    PyDict_SetItemString(py_time, "solve",
                         PyFloat_FromDouble(time->solve));
    PyDict_SetItemString(py_time, "clock_total",
                         PyFloat_FromDouble(time->clock_total));
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
PyObject* dps_make_inform_dict(const struct dps_inform_type *inform){
    PyObject *py_inform = PyDict_New();


    PyDict_SetItemString(py_inform, "status",
                         PyLong_FromLong(inform->status));
    PyDict_SetItemString(py_inform, "alloc_status",
                         PyLong_FromLong(inform->alloc_status));
    PyDict_SetItemString(py_inform, "mod_1by1",
                         PyLong_FromLong(inform->mod_1by1));
    PyDict_SetItemString(py_inform, "mod_2by2",
                         PyLong_FromLong(inform->mod_2by2));
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
    PyDict_SetItemString(py_inform, "hard_case",
                         PyBool_FromLong(inform->hard_case));
    PyDict_SetItemString(py_inform, "bad_alloc",
                         PyUnicode_FromString(inform->bad_alloc));

    // Set time nested dictionary
    PyDict_SetItemString(py_inform, "time",
                         dps_make_time_dict(&inform->time));

    // Set dictionaries from subservient packages
    PyDict_SetItemString(py_inform, "sls_inform",
                         sls_make_inform_dict(&inform->sls_inform));

    return py_inform;
}

//  *-*-*-*-*-*-*-*-*-*-   DPS_INITIALIZE    -*-*-*-*-*-*-*-*-*-*

static PyObject* py_dps_initialize(PyObject *self){

    // Call dps_initialize
    dps_initialize(&data, &control, &status);

    // Record that dps has been initialised
    init_called = true;

    // Return options Python dictionary
    PyObject *py_options = dps_make_options_dict(&control);
    return Py_BuildValue("O", py_options);
}

//  *-*-*-*-*-*-*-*-*-*-*-*-   DPS_LOAD    -*-*-*-*-*-*-*-*-*-*-*-*

static PyObject* py_dps_load(PyObject *self, PyObject *args, PyObject *keywds){
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
    dps_reset_control(&control, &data, &status);

    // Update dps control options
    if(!dps_update_control(&control, py_options))
        return NULL;

    // Call dps_import
    dps_import(&control, &data, &status, n,
               H_type, H_ne, H_row, H_col, H_ptr);

    // Free allocated memory
    if(H_row != NULL) free(H_row);
    if(H_col != NULL) free(H_col);
    if(H_ptr != NULL) free(H_ptr);

    // Raise any status errors
    if(!check_error_codes(status))
        return NULL;

    // Record that dps structures been initialised
    load_called = true;

    // Return None boilerplate
    Py_INCREF(Py_None);
    return Py_None;
}

//  *-*-*-*-*-*-*-*-*-*-   DPS_SOLVE_TR_PROBLEM   -*-*-*-*-*-*-*-*

static PyObject* py_dps_solve_tr_problem(PyObject *self, PyObject *args, PyObject *keywds){
    PyArrayObject *py_g, *py_H_val;
    double *g, *H_val;
    int n, H_ne;
    double radius, f;

    // Check that package has been initialised
    if(!check_load(load_called))
        return NULL;

    // Parse positional arguments
    static char *kwlist[] = {"n","radius","f","g","H_ne","H_val",NULL};
    if(!PyArg_ParseTupleAndKeywords(args, keywds, "iddOiO", kwlist, &n, &radius, &f, &py_g,
                                    &H_ne, &py_H_val))
        return NULL;

    // Check that array inputs are of correct type, size, and shape
    if(!check_array_double("g", py_g, n))
        return NULL;
    if(!check_array_double("H_val", py_H_val, H_ne))
        return NULL;

    // Get array data pointer
    g = (double *) PyArray_DATA(py_g);
    H_val = (double *) PyArray_DATA(py_H_val);

   // Create NumPy output arrays
    npy_intp ndim[] = {n}; // size of x
    PyArrayObject *py_x =
      (PyArrayObject *) PyArray_SimpleNew(1, ndim, NPY_DOUBLE);
    double *x = (double *) PyArray_DATA(py_x);

    // Call dps_solve_problem
    dps_solve_tr_problem(&data, &status, n, H_ne, H_val, g, f, radius, x);
    // for( int i = 0; i < n; i++) printf("x %f\n", x[i]);

    // Propagate any errors with the callback function
    if(PyErr_Occurred())
        return NULL;

    // Raise any status errors
    if(!check_error_codes(status))
        return NULL;

    // Return x
    PyObject *solve_tr_problem_return;

    solve_tr_problem_return = Py_BuildValue("O", py_x);
    Py_INCREF(solve_tr_problem_return);
    return solve_tr_problem_return;

}

//  *-*-*-*-*-*-*-*-*-*-   DPS_SOLVE_RQ_PROBLEM   -*-*-*-*-*-*-*-*

static PyObject* py_dps_solve_rq_problem(PyObject *self, PyObject *args, PyObject *keywds){
    PyArrayObject *py_g, *py_H_val;
    double *g, *H_val;
    int n, H_ne;
    double power, weight, f;

    // Check that package has been initialised
    if(!check_load(load_called))
        return NULL;

    // Parse positional arguments
    static char *kwlist[] = {"n","weight","power","f","g","H_ne","H_val",NULL};
    if(!PyArg_ParseTupleAndKeywords(args, keywds, "idddOiO", kwlist, &n, 
                                    &weight, &power, &f, &py_g,
                                    &H_ne, &py_H_val))
        return NULL;

    // Check that array inputs are of correct type, size, and shape
    if(!check_array_double("g", py_g, n))
        return NULL;
    if(!check_array_double("H_val", py_H_val, H_ne))
        return NULL;

    // Get array data pointer
    g = (double *) PyArray_DATA(py_g);
    H_val = (double *) PyArray_DATA(py_H_val);

   // Create NumPy output arrays
    npy_intp ndim[] = {n}; // size of x
    PyArrayObject *py_x =
      (PyArrayObject *) PyArray_SimpleNew(1, ndim, NPY_DOUBLE);
    double *x = (double *) PyArray_DATA(py_x);

    // Call dps_solve_problem
    dps_solve_rq_problem(&data, &status, n, H_ne, H_val, g, f,
                         weight, power, x);
    // for( int i = 0; i < n; i++) printf("x %f\n", x[i]);

    // Propagate any errors with the callback function
    if(PyErr_Occurred())
        return NULL;

    // Raise any status errors
    if(!check_error_codes(status))
        return NULL;

    // Return x
    PyObject *solve_rq_problem_return;

    solve_rq_problem_return = Py_BuildValue("O", py_x);
    Py_INCREF(solve_rq_problem_return);
    return solve_rq_problem_return;

}

//  *-*-*-*-*-*-*-*-*-*-   DPS_INFORMATION   -*-*-*-*-*-*-*-*

static PyObject* py_dps_information(PyObject *self){

    // Check that package has been initialised
    if(!check_init(init_called))
        return NULL;

    // Call dps_information
    dps_information(&data, &inform, &status);

    // Return status and inform Python dictionary
    PyObject *py_inform = dps_make_inform_dict(&inform);
    return Py_BuildValue("O", py_inform);
}

//  *-*-*-*-*-*-*-*-*-*-   DPS_TERMINATE   -*-*-*-*-*-*-*-*-*-*

static PyObject* py_dps_terminate(PyObject *self){

    // Check that package has been initialised
    if(!check_init(init_called))
        return NULL;

    // Call dps_terminate
    dps_terminate(&data, &control, &inform);

    // require future calls start with dps_initialize
    init_called = false;
    load_called = false;

    // Return None boilerplate
    Py_INCREF(Py_None);
    return Py_None;
}

//  *-*-*-*-*-*-*-*-*-*-   INITIALIZE DPS PYTHON MODULE    -*-*-*-*-*-*-*-*-*-*

/* dps python module method table */
static PyMethodDef dps_module_methods[] = {
    {"initialize", (PyCFunction) py_dps_initialize, METH_NOARGS, NULL},
    {"load", (PyCFunction) py_dps_load, METH_VARARGS | METH_KEYWORDS, NULL},
    {"solve_tr_problem", (PyCFunction) py_dps_solve_tr_problem,
     METH_VARARGS | METH_KEYWORDS, NULL},
    {"solve_rq_problem", (PyCFunction) py_dps_solve_rq_problem,
     METH_VARARGS | METH_KEYWORDS, NULL},
    {"information", (PyCFunction) py_dps_information, METH_NOARGS, NULL},
    {"terminate", (PyCFunction) py_dps_terminate, METH_NOARGS, NULL},
    {NULL, NULL, 0, NULL}  /* Sentinel */
};

/* dps python module documentation */

PyDoc_STRVAR(dps_module_doc,

"The dps package constructs a symmetric, positive definite matrix M \n"
"from a given H so that H is is diagonal \n"
"in the norm ||v||_M = sqrt{v^T M v} induced by M, and consequently \n"
"minimizers of trust-region and regularized quadratic subproblems \n"
"may be computed efficiently.\n"
"The aim is either to minimize the quadratic objective function\n"
"q(x) = f + g^T x + 1/2 x^T H x, \n"
"where the vector x is required to satisfy \n"
"the ellipsoidal  trust-region constraint ||x||_M <= Delta, \n"
"or to minimize the regularized quadratic objective\n"
"r(x) = q(x) + sigma/p ||x||_M^p,\n"
"where the radius Delta > 0, the weight sigma > 0, and the power p >= 2.\n"
"A factorization of the matrix H will be required, so this package is\n"
"most suited for the case where such a factorization,\n"
"either dense or sparse, may be found efficiently.\n"
"\n"
"See $GALAHAD/html/Python/dps.html for argument lists, call order\n"
"and other details.\n"
"\n"
);

/* dps python module definition */
static struct PyModuleDef module = {
   PyModuleDef_HEAD_INIT,
   "dps",               /* name of module */
   dps_module_doc,      /* module documentation, may be NULL */
   -1,                  /* size of per-interpreter state of the module,or -1
                           if the module keeps state in global variables */
   dps_module_methods   /* module methods */
};

/* Python module initialization */
PyMODINIT_FUNC PyInit_dps(void) { // must be same as module name above
    import_array();  // for NumPy arrays
    return PyModule_Create(&module);
}
