//* \file sha_pyiface.c */

/*
 * THIS VERSION: GALAHAD 5.1 - 2024-11-05 AT 14:20 GMT.
 *
 *-*-*-*-*-*-*-*-*-  GALAHAD_SHA PYTHON INTERFACE  *-*-*-*-*-*-*-*-*-*-
 *
 *  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
 *  Principal author: Jaroslav Fowkes & Nick Gould
 *
 *  History -
 *   originally released GALAHAD Version 4.1. May 3rd 2023
 *
 *  For full documentation, see
 *   http://galahad.rl.ac.uk/galahad-www/specs.html
 */

#include "galahad_python.h"
#include "galahad_sha.h"

/* Module global variables */
static void *data;                       // private internal data
static struct sha_control_type control;  // control struct
static struct sha_inform_type inform;    // inform struct
static bool init_called = false;         // record if initialise was called
static int status = 0;                   // exit status

//  *-*-*-*-*-*-*-*-*-*-   UPDATE CONTROL    -*-*-*-*-*-*-*-*-*-*

/* Update the control options: use C defaults but update any passed via Python*/
// NB not static as it is used for nested control within TRU Python interface
bool sha_update_control(struct sha_control_type *control,
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
        if(strcmp(key_name, "approximation_algorithm") == 0){
            if(!parse_int_option(value, "approximation_algorithm",
                                  &control->approximation_algorithm))
                return false;
            continue;
        }
        if(strcmp(key_name, "dense_linear_solver") == 0){
            if(!parse_int_option(value, "dense_linear_solver",
                                  &control->dense_linear_solver))
                return false;
            continue;
        }
        if(strcmp(key_name, "extra_differences") == 0){
            if(!parse_int_option(value, "extra_differences",
                                  &control->extra_differences))
                return false;
            continue;
        }
        if(strcmp(key_name, "sparse_row") == 0){
            if(!parse_int_option(value, "sparse_row",
                                  &control->sparse_row))
                return false;
            continue;
        }
        if(strcmp(key_name, "recursion_max") == 0){
            if(!parse_int_option(value, "recursion_max",
                                  &control->recursion_max))
                return false;
            continue;
        }
        if(strcmp(key_name, "recursion_entries_required") == 0){
            if(!parse_int_option(value, "recursion_entries_required",
                                  &control->recursion_entries_required))
                return false;
            continue;
        }
        if(strcmp(key_name, "average_off_diagonals") == 0){
            if(!parse_bool_option(value, "average_off_diagonals",
                                  &control->average_off_diagonals))
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
PyObject* sha_make_options_dict(const struct sha_control_type *control){
    PyObject *py_options = PyDict_New();

    PyDict_SetItemString(py_options, "error",
                         PyLong_FromLong(control->error));
    PyDict_SetItemString(py_options, "out",
                         PyLong_FromLong(control->out));
    PyDict_SetItemString(py_options, "print_level",
                         PyLong_FromLong(control->print_level));
    PyDict_SetItemString(py_options, "approximation_algorithm",
                         PyLong_FromLong(control->approximation_algorithm));
    PyDict_SetItemString(py_options, "dense_linear_solver",
                         PyLong_FromLong(control->dense_linear_solver));
    PyDict_SetItemString(py_options, "extra_differences",
                         PyLong_FromLong(control->extra_differences));
    PyDict_SetItemString(py_options, "sparse_row",
                         PyLong_FromLong(control->sparse_row));
    PyDict_SetItemString(py_options, "recursion_max",
                         PyLong_FromLong(control->recursion_max));
    PyDict_SetItemString(py_options, "recursion_entries_required",
                         PyLong_FromLong(control->recursion_entries_required));
    PyDict_SetItemString(py_options, "average_off_diagonals",
                         PyBool_FromLong(control->average_off_diagonals));
    PyDict_SetItemString(py_options, "space_critical",
                         PyBool_FromLong(control->space_critical));
    PyDict_SetItemString(py_options, "deallocate_error_fatal",
                         PyBool_FromLong(control->deallocate_error_fatal));
    PyDict_SetItemString(py_options, "prefix",
                         PyUnicode_FromString(control->prefix));
    return py_options;
}

//  *-*-*-*-*-*-*-*-*-*-   MAKE INFORM    -*-*-*-*-*-*-*-*-*-*

/* Take the inform struct from C and turn it into a python dictionary */
// NB not static as it is used for nested control within TRU Python interface
PyObject* sha_make_inform_dict(const struct sha_inform_type *inform){
    PyObject *py_inform = PyDict_New();

    PyDict_SetItemString(py_inform, "status",
                         PyLong_FromLong(inform->status));
    PyDict_SetItemString(py_inform, "alloc_status",
                         PyLong_FromLong(inform->alloc_status));
    PyDict_SetItemString(py_inform, "max_degree",
                         PyLong_FromLong(inform->max_degree));
    PyDict_SetItemString(py_inform, "differences_needed",
                         PyLong_FromLong(inform->differences_needed));
    PyDict_SetItemString(py_inform, "max_reduced_degree",
                         PyLong_FromLong(inform->max_reduced_degree));
    PyDict_SetItemString(py_inform, "bad_row",
                         PyLong_FromLong(inform->bad_row));
    PyDict_SetItemString(py_inform, "approximation_algorithm_used",
                         PyLong_FromLong(inform->approximation_algorithm_used));
    PyDict_SetItemString(py_inform, "max_off_diagonal_difference",
                         PyFloat_FromDouble(inform->max_off_diagonal_difference));
    PyDict_SetItemString(py_inform, "bad_alloc",
                         PyUnicode_FromString(inform->bad_alloc));
    return py_inform;
}

//  *-*-*-*-*-*-*-*-*-*-   SHA_INITIALIZE    -*-*-*-*-*-*-*-*-*-*

static PyObject* py_sha_initialize(PyObject *self){

    // Call sha_initialize
    sha_initialize(&data, &control, &status);

    // Record that SHA has been initialised
    init_called = true;

    // Return options Python dictionary
    PyObject *py_options = sha_make_options_dict(&control);
    return Py_BuildValue("O", py_options);
}

//  *-*-*-*-*-*-*-*-*-*-*-*-   SHA_ANALYSE_MATRIX    -*-*-*-*-*-*-*-*-*-*-*-*

static PyObject* py_sha_analyse_matrix(PyObject *self, PyObject *args, PyObject *keywds){
    PyArrayObject *py_row, *py_col;
    PyObject *py_options = NULL;
    int *row = NULL, *col = NULL;
    int n, ne, m;

    // Check that package has been initialised
    if(!check_init(init_called))
        return NULL;

    // Parse positional and keyword arguments
    static char *kwlist[] = {"n","ne","row","col","options",NULL};
    if(!PyArg_ParseTupleAndKeywords(args, keywds, "iiOO|O",
                                    kwlist, &n, &ne, &py_row, &py_col,
                                    &py_options))
        return NULL;

    // Check that array inputs are of correct type, size, and shape

    if(!(
        check_array_int("row", py_row, ne) &&
        check_array_int("col", py_col, ne)
        ))
        return NULL;

    // Convert 64bit integer row array to 32bit
    row = malloc(ne * sizeof(int));
    long int *row_long = (long int *) PyArray_DATA(py_row);
    for(int i = 0; i < ne; i++) row[i] = (int) row_long[i];

    // Convert 64bit integer col array to 32bit
    col = malloc(ne * sizeof(int));
    long int *col_long = (long int *) PyArray_DATA(py_col);
    for(int i = 0; i < ne; i++) col[i] = (int) col_long[i];

    // Reset control options
    sha_reset_control(&control, &data, &status);

    // Update SHA control options
    if(!sha_update_control(&control, py_options))
        return NULL;

    // Call sha_analyse_matrix
    sha_analyse_matrix(&control, &data, &status, n, ne, row, col, &m);

    // Free allocated memory
    free(row);
    free(col);

    // Raise any status errors
    if(!check_error_codes(status))
        return NULL;

    // Return m
    PyObject *analyse_matrix_return;
    analyse_matrix_return = Py_BuildValue("i", m);
    Py_INCREF(analyse_matrix_return);
    return analyse_matrix_return;

}

//  *-*-*-*-*-*-*-*-*-*-*-*-   SHA_RECOVER_MATRIX   -*-*-*-*-*-*-*-*-*

static PyObject* py_sha_recover_matrix(PyObject *self, PyObject *args, PyObject *keywds){
    PyArrayObject *py_order = NULL;
    PyArrayObject *py_s1, *py_y1;
    int ne, m, ls1, ls2, ly1, ly2;
    int *order = NULL;

    // Check that package has been initialised
    if(!check_init(init_called))
        return NULL;

    // Parse positional and keyword arguments
    static char *kwlist[] = {"ne","m","ls1","ls2","strans","ly1","ly2","ytrans",
                             "order",NULL};

    if(!PyArg_ParseTupleAndKeywords(args, keywds, "iiiiOiiO|O",
                                    kwlist, &ne, &m, &ls1, &ls2, &py_s1,
                                    &ly1, &ly2, &py_y1, &py_order))
        return NULL;

    // s and y are actually 2d arrays, so make the connection
    npy_intp sdims[] = {ls1, ls2};
    npy_intp ydims[] = {ly1, ly2};
    PyArrayObject *py_s = (PyArrayObject *) PyArray_SimpleNew(2, sdims,
                                                              NPY_DOUBLE);
    PyArrayObject *py_y = (PyArrayObject *) PyArray_SimpleNew(2, ydims,
                                                              NPY_DOUBLE);
    py_s = py_s1;
    py_y = py_y1;

    // Check that array inputs are of correct type, size, and shape
    if(!check_2darray_double("s", py_s, ls1, ls2))
        return NULL;
    if(!check_2darray_double("y", py_y, ly1, ly2))
        return NULL;

    // Convert 64bit integer order to 32bit
    if((PyObject *) py_order != NULL){
        if(!check_array_int("order", py_order, m))
            return NULL;
        order = malloc(m * sizeof(int));
        long int *order_long = (long int *) PyArray_DATA(py_order);
        for(int i = 0; i < m; i++) order[i] = (int) order_long[i];
    }

    // Get array data pointers
    double *s = (double *) PyArray_DATA(py_s);
    double (*s2d)[ls2] = (double (*)[ls2]) s;
    double *y = (double *) PyArray_DATA(py_y);
    double (*y2d)[ly2] = (double (*)[ly2]) y;

    //for( int i = 0; i < ls2; i++) printf("s %f\n", s2d[0][i]);

   // Create NumPy output array for val
    npy_intp nedim[] = {ne};
    PyArrayObject *py_val =
      (PyArrayObject *) PyArray_SimpleNew(1, nedim, NPY_DOUBLE);
    double *val = (double *) PyArray_DATA(py_val);

    // Call sha_solve_direct
    sha_recover_matrix( &data, &status, ne, m, ls1, ls2, s2d, ly1, ly2, y2d,
                        val, order );

    //for( int i = 0; i < ne; i++) printf("val %f\n", val[i]);

    // Free allocated memory
    if(order != NULL) free(order);

    // Propagate any errors with the callback function
    if(PyErr_Occurred())
        return NULL;

    // Raise any status errors
    if(!check_error_codes(status))
        return NULL;

    // Return val
    PyObject *recover_matrix_return;
    recover_matrix_return = Py_BuildValue("O", py_val);
    Py_INCREF(recover_matrix_return);
    return recover_matrix_return;

}

//  *-*-*-*-*-*-*-*-*-*-   SHA_INFORMATION   -*-*-*-*-*-*-*-*

static PyObject* py_sha_information(PyObject *self){

    // Check that package has been initialised
    if(!check_init(init_called))
        return NULL;

    // Call sha_information
    sha_information(&data, &inform, &status);

    // Return status and inform Python dictionary
    PyObject *py_inform = sha_make_inform_dict(&inform);
    return Py_BuildValue("O", py_inform);
}

//  *-*-*-*-*-*-*-*-*-*-   SHA_TERMINATE   -*-*-*-*-*-*-*-*-*-*

static PyObject* py_sha_terminate(PyObject *self){

    // Check that package has been initialised
    if(!check_init(init_called))
        return NULL;

    // Call sha_terminate
    sha_terminate(&data, &control, &inform);

    // Return None boilerplate
    Py_INCREF(Py_None);
    return Py_None;
}

//  *-*-*-*-*-*-*-*-*-*-   INITIALIZE SHA PYTHON MODULE    -*-*-*-*-*-*-*-*-*-*

/* sha python module method table */
static PyMethodDef sha_module_methods[] = {
    {"initialize", (PyCFunction) py_sha_initialize, METH_NOARGS,NULL},
    {"analyse_matrix", (PyCFunction) py_sha_analyse_matrix,
       METH_VARARGS | METH_KEYWORDS, NULL},
    {"recover_matrix", (PyCFunction) py_sha_recover_matrix,
       METH_VARARGS | METH_KEYWORDS, NULL},
    {"information", (PyCFunction) py_sha_information, METH_NOARGS, NULL},
    {"terminate", (PyCFunction) py_sha_terminate, METH_NOARGS, NULL},
    {NULL, NULL, 0, NULL}  /* Sentinel */
};

/* sha python module documentation */

PyDoc_STRVAR(sha_module_doc,

"The sha package finds an approximation to a sparse Hessian \n"
"using componentwise secant approximation.\n"
"\n"
"See $GALAHAD/html/Python/sha.html for argument lists, call order\n"
"and other details.\n"
"\n"
);

/* sha python module definition */
static struct PyModuleDef module = {
   PyModuleDef_HEAD_INIT,
   "sha",               /* name of module */
   sha_module_doc,      /* module documentation, may be NULL */
   -1,                  /* size of per-interpreter state of the module,or -1
                           if the module keeps state in global variables */
   sha_module_methods   /* module methods */
};

/* Python module initialization */
PyMODINIT_FUNC PyInit_sha(void) { // must be same as module name above
    import_array();  // for NumPy arrays
    return PyModule_Create(&module);
}
