//* \file fdc_pyiface.c */

/*
 * THIS VERSION: GALAHAD 4.1 - 2023-05-20 AT 10:30 GMT.
 *
 *-*-*-*-*-*-*-*-*-  GALAHAD_FDC PYTHON INTERFACE  *-*-*-*-*-*-*-*-*-*-
 *
 *  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
 *  Principal author: Jaroslav Fowkes & Nick Gould
 *
 *  History -
 *   originally released GALAHAD Version 4.1. April 2nd 2023
 *
 *  For full documentation, see
 *   http://galahad.rl.ac.uk/galahad-www/specs.html
 */

#include "galahad_python.h"
#include "galahad_fdc.h"

/* Nested GALAHAD control/inform prototypes */
bool sls_update_control(struct sls_control_type *control,
                        PyObject *py_options);
PyObject* sls_make_options_dict(const struct sls_control_type *control);
PyObject* sls_make_inform_dict(const struct sls_inform_type *inform);
bool uls_update_control(struct uls_control_type *control,
                        PyObject *py_options);
PyObject* uls_make_options_dict(const struct uls_control_type *control);
PyObject* uls_make_inform_dict(const struct uls_inform_type *inform);

/* Module global variables */
static void *data;                       // private internal data
static struct fdc_control_type control;  // control struct
static struct fdc_inform_type inform;    // inform struct
static bool init_called = false;         // record if initialise was called
static int status = 0;                   // exit status

//  *-*-*-*-*-*-*-*-*-*-   UPDATE CONTROL    -*-*-*-*-*-*-*-*-*-*

/* Update the control options: use C defaults but update any passed via Python*/
// NB not static as it is used for nested control within SBLS Python interface
bool fdc_update_control(struct fdc_control_type *control,
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
        if(strcmp(key_name, "pivot_tol") == 0){
            if(!parse_double_option(value, "pivot_tol",
                                  &control->pivot_tol))
                return false;
            continue;
        }
        if(strcmp(key_name, "zero_pivot") == 0){
            if(!parse_double_option(value, "zero_pivot",
                                  &control->zero_pivot))
                return false;
            continue;
        }
        if(strcmp(key_name, "max_infeas") == 0){
            if(!parse_double_option(value, "max_infeas",
                                  &control->max_infeas))
                return false;
            continue;
        }
        if(strcmp(key_name, "use_sls") == 0){
            if(!parse_bool_option(value, "use_sls",
                                  &control->use_sls))
                return false;
            continue;
        }
        if(strcmp(key_name, "scale") == 0){
            if(!parse_bool_option(value, "scale",
                                  &control->scale))
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
        if(strcmp(key_name, "symmetric_linear_solver") == 0){
            if(!parse_char_option(value, "symmetric_linear_solver",
                                  control->symmetric_linear_solver,
                                  sizeof(control->symmetric_linear_solver)))
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
        if(strcmp(key_name, "uls_options") == 0){
            if(!uls_update_control(&control->uls_control, value))
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
// NB not static as it is used for nested inform within SBLS Python interface
PyObject* fdc_make_options_dict(const struct fdc_control_type *control){
    PyObject *py_options = PyDict_New();

    PyDict_SetItemString(py_options, "error",
                         PyLong_FromLong(control->error));
    PyDict_SetItemString(py_options, "out",
                         PyLong_FromLong(control->out));
    PyDict_SetItemString(py_options, "print_level",
                         PyLong_FromLong(control->print_level));
    PyDict_SetItemString(py_options, "indmin",
                         PyLong_FromLong(control->indmin));
    PyDict_SetItemString(py_options, "valmin",
                         PyLong_FromLong(control->valmin));
    PyDict_SetItemString(py_options, "pivot_tol",
                         PyFloat_FromDouble(control->pivot_tol));
    PyDict_SetItemString(py_options, "zero_pivot",
                         PyFloat_FromDouble(control->zero_pivot));
    PyDict_SetItemString(py_options, "max_infeas",
                         PyFloat_FromDouble(control->max_infeas));
    PyDict_SetItemString(py_options, "use_sls",
                         PyBool_FromLong(control->use_sls));
    PyDict_SetItemString(py_options, "scale",
                         PyBool_FromLong(control->scale));
    PyDict_SetItemString(py_options, "space_critical",
                         PyBool_FromLong(control->space_critical));
    PyDict_SetItemString(py_options, "deallocate_error_fatal",
                         PyBool_FromLong(control->deallocate_error_fatal));
    PyDict_SetItemString(py_options, "symmetric_linear_solver",
                         PyUnicode_FromString(control->symmetric_linear_solver));
    PyDict_SetItemString(py_options, "unsymmetric_linear_solver",
                         PyUnicode_FromString(control->unsymmetric_linear_solver));
    PyDict_SetItemString(py_options, "prefix",
                         PyUnicode_FromString(control->prefix));
    PyDict_SetItemString(py_options, "sls_options",
                         sls_make_options_dict(&control->sls_control));
    PyDict_SetItemString(py_options, "uls_options",
                         uls_make_options_dict(&control->uls_control));

    return py_options;
}

//  *-*-*-*-*-*-*-*-*-*-   MAKE TIME    -*-*-*-*-*-*-*-*-*-*

/* Take the time struct from C and turn it into a python dictionary */
static PyObject* fdc_make_time_dict(const struct fdc_time_type *time){
    PyObject *py_time = PyDict_New();

// Set float/double time entries
    PyDict_SetItemString(py_time, "total",
                         PyFloat_FromDouble(time->total));
    PyDict_SetItemString(py_time, "analyse",
                         PyFloat_FromDouble(time->analyse));
    PyDict_SetItemString(py_time, "factorize",
                         PyFloat_FromDouble(time->factorize));
    PyDict_SetItemString(py_time, "clock_total",
                         PyFloat_FromDouble(time->clock_total));
    PyDict_SetItemString(py_time, "clock_analyse",
                         PyFloat_FromDouble(time->clock_analyse));
    PyDict_SetItemString(py_time, "clock_factorize",
                         PyFloat_FromDouble(time->clock_factorize));

     return py_time;
}

//  *-*-*-*-*-*-*-*-*-*-   MAKE INFORM    -*-*-*-*-*-*-*-*-*-*

/* Take the inform struct from C and turn it into a python dictionary */
// NB not static as it is used for nested control within SBLS Python interface
PyObject* fdc_make_inform_dict(const struct fdc_inform_type *inform){
    PyObject *py_inform = PyDict_New();

    PyDict_SetItemString(py_inform, "status",
                         PyLong_FromLong(inform->status));
    PyDict_SetItemString(py_inform, "alloc_status",
                         PyLong_FromLong(inform->alloc_status));
    PyDict_SetItemString(py_inform, "bad_alloc",
                         PyUnicode_FromString(inform->bad_alloc));
    PyDict_SetItemString(py_inform, "factorization_status",
                         PyLong_FromLong(inform->factorization_status));
    PyDict_SetItemString(py_inform, "factorization_integer",
                         PyLong_FromLong(inform->factorization_integer));
    PyDict_SetItemString(py_inform, "factorization_real",
                         PyLong_FromLong(inform->factorization_real));
    PyDict_SetItemString(py_inform, "non_negligible_pivot",
                         PyFloat_FromDouble(inform->non_negligible_pivot));
    PyDict_SetItemString(py_inform, "sls_inform",
                         sls_make_inform_dict(&inform->sls_inform));
    PyDict_SetItemString(py_inform, "uls_inform",
                         uls_make_inform_dict(&inform->uls_inform));
    // Set time nested dictionary
    PyDict_SetItemString(py_inform, "time",
                         fdc_make_time_dict(&inform->time));

    return py_inform;
}

//  *-*-*-*-*-*-*-*-*-*-   FDC_INITIALIZE    -*-*-*-*-*-*-*-*-*-*

static PyObject* py_fdc_initialize(PyObject *self){

    // Call fdc_initialize
    fdc_initialize(&data, &control, &status);

    // Record that FDC has been initialised
    init_called = true;

    // Return options Python dictionary
    PyObject *py_options = fdc_make_options_dict(&control);
    return Py_BuildValue("O", py_options);
}

//  *-*-*-*-*-*-*-*-*-*-*-*-   FDC_FACTORIZE_MATRIX    -*-*-*-*-*-*-*-*-*-*-*-*

static PyObject* py_fdc_find_dependent_rows(PyObject *self, PyObject *args,
                                            PyObject *keywds){
    PyArrayObject *py_A_val, *py_A_col, *py_A_ptr, *py_b;
    PyObject *py_options = NULL;
    int *A_col = NULL, *A_ptr = NULL;
    double *A_val, *b;
    int m, n, A_ne, n_depen;

    // Check that package has been initialised
    if(!check_init(init_called))
        return NULL;

    // Parse positional and keyword arguments
    static char *kwlist[] = {"m","n","A_ptr","A_col","A_val","b",
                             "options",NULL};

    if(!PyArg_ParseTupleAndKeywords(args, keywds, "iiOOOO|O", kwlist,
                                    &m, &n, &py_A_ptr, &py_A_col, &py_A_val,
                                    &py_b, &py_options))
        return NULL;

    // Check that rpw pointer array is of correct type, size, and shape

    if(!(check_array_int("A_ptr", py_A_ptr, m+1)))
        return NULL;

    // Convert 64bit integer A_ptr array to 32bit
    A_ptr = malloc((n+1) * sizeof(int));
    long int *A_ptr_long = (long int *) PyArray_DATA(py_A_ptr);
    for(int i = 0; i < m+1; i++) A_ptr[i] = (int) A_ptr_long[i];
    A_ne = A_ptr[m];

    // Check that remaining array inputs are of correct type, size, and shape

    if(!(
        check_array_int("A_col", py_A_col, A_ne) &&
        check_array_double("A_val", py_A_val, A_ne) &&
        check_array_double("b", py_b, m)
        ))
        return NULL;

    // Convert 64bit integer A_col array to 32bit
    A_col = malloc(A_ne * sizeof(int));
    long int *A_col_long = (long int *) PyArray_DATA(py_A_col);
    for(int i = 0; i < A_ne; i++) A_col[i] = (int) A_col_long[i];

    A_val = (double *) PyArray_DATA(py_A_val);
    b = (double *) PyArray_DATA(py_b);

    // Reset control options
    //fdc_reset_control(&control, &data, &status);

    // Update FDC control options
    if(!fdc_update_control(&control, py_options))
        return NULL;

    // Create NumPy output array
    npy_intp ndim[] = {n}; // size of dimen
    PyArrayObject *py_depen =
      (PyArrayObject *) PyArray_SimpleNew(1, ndim, NPY_INT);
    int *depen = (int *) PyArray_DATA(py_depen);

    // Call fdc_find_dependent_rows
    fdc_find_dependent_rows(&control, &data, &inform, &status, m, n,
                            A_ne, A_col, A_ptr, A_val, b, &n_depen, depen);
    //printf("A_ne %i\n", A_ne);
    //printf("n_depen %i\n", n_depen);

    // Propagate any errors with the callback function
    if(PyErr_Occurred())
        return NULL;

    // Raise any status errors
    if(!check_error_codes(status))
        return NULL;

    // Free allocated memory
    free(A_col);
    free(A_ptr);

    // Return n_depen, depen and inform
    PyObject *find_dependent_rows_return;
    PyObject *py_inform = fdc_make_inform_dict(&inform);
    find_dependent_rows_return =
      Py_BuildValue("iOO", n_depen, py_depen, py_inform);
    Py_INCREF(find_dependent_rows_return);
    return find_dependent_rows_return;
}

//  *-*-*-*-*-*-*-*-*-*-   FDC_TERMINATE   -*-*-*-*-*-*-*-*-*-*

static PyObject* py_fdc_terminate(PyObject *self){

    // Check that package has been initialised
    if(!check_init(init_called))
        return NULL;

    // Call fdc_terminate
    fdc_terminate(&data, &control, &inform);

    // Record that FDC must be reinitialised if called again
    init_called = false;

    // Return None boilerplate
    Py_INCREF(Py_None);
    return Py_None;
}

//  *-*-*-*-*-*-*-*-*-*-   INITIALIZE FDC PYTHON MODULE    -*-*-*-*-*-*-*-*-*-*

/* fdc python module method table */
static PyMethodDef fdc_module_methods[] = {
    {"initialize", (PyCFunction) py_fdc_initialize, METH_NOARGS, NULL},
    {"find_dependent_rows", (PyCFunction) py_fdc_find_dependent_rows,
     METH_VARARGS | METH_KEYWORDS, NULL},
    {"terminate", (PyCFunction) py_fdc_terminate, METH_NOARGS, NULL},
    {NULL, NULL, 0, NULL}  /* Sentinel */
};

/* fdc python module documentation */

PyDoc_STRVAR(fdc_module_doc,

"Given an under-determined set of linear equations/constraints \n"
"a_i^T x = b_i, i = 1, ..., m involving n >= m unknowns x, the fdc \n"
"package determines whether the constraints are consistent, and if \n"
"so how many of the constraints are dependent; a list of dependent \n"
"constraints, that is, those which may be removed without changing the \n"
"solution set, will be found and the remaining a_i will be linearly \n"
"independent.  Full advantage is taken of any zero coefficients in the \n"
"matrix A whose columns are the vectors a_i^T.\n"
"\n"
"See $GALAHAD/html/Python/fdc.html for argument lists, call order\n"
"and other details.\n"
"\n"
);

/* fdc python module definition */
static struct PyModuleDef module = {
   PyModuleDef_HEAD_INIT,
   "fdc",               /* name of module */
   fdc_module_doc,      /* module documentation, may be NULL */
   -1,                  /* size of per-interpreter state of the module,or -1
                           if the module keeps state in global variables */
   fdc_module_methods   /* module methods */
};

/* Python module initialization */
PyMODINIT_FUNC PyInit_fdc(void) { // must be same as module name above
    import_array();  // for NumPy arrays
    return PyModule_Create(&module);
}
