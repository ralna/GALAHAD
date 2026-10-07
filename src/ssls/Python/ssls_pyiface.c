//* \file ssls_pyiface.c */

/*
 * THIS VERSION: GALAHAD 5.3 - 2025-07-27 AT 08:20 GMT.
 *
 *-*-*-*-*-*-*-*-*-  GALAHAD_SSLS PYTHON INTERFACE  *-*-*-*-*-*-*-*-*-*-
 *
 *  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
 *  Principal author: Jaroslav Fowkes & Nick Gould
 *
 *  History -
 *   originally released GALAHAD Version 5.3. July 27th 2023
 *
 *  For full documentation, see
 *   http://galahad.rl.ac.uk/galahad-www/specs.html
 */

#include "galahad_python.h"
#include "galahad_ssls.h"

/* Nested HSL info/inform prototypes */
bool sls_update_control(struct sls_control_type *control,
                        PyObject *py_options);
PyObject* sls_make_options_dict(const struct sls_control_type *control);
PyObject* sls_make_inform_dict(const struct sls_inform_type *inform);

/* Module global variables */
static void *data;                       // private internal data
static struct ssls_control_type control;  // control struct
static struct ssls_inform_type inform;    // inform struct
static bool init_called = false;         // record if initialise was called
static int status = 0;                   // exit status

//  *-*-*-*-*-*-*-*-*-*-   UPDATE CONTROL    -*-*-*-*-*-*-*-*-*-*

/* Update the control options: use C defaults but update any passed via Python*/
// NB not static as it is used for nested control within SSLS Python interface
bool ssls_update_control(struct ssls_control_type *control,
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
// NB not static as it is used for nested inform within CQP Python interface
PyObject* ssls_make_options_dict(const struct ssls_control_type *control){
    PyObject *py_options = PyDict_New();

    PyDict_SetItemString(py_options, "error",
                         PyLong_FromLong(control->error));
    PyDict_SetItemString(py_options, "out",
                         PyLong_FromLong(control->out));
    PyDict_SetItemString(py_options, "print_level",
                         PyLong_FromLong(control->print_level));
    PyDict_SetItemString(py_options, "space_critical",
                         PyBool_FromLong(control->space_critical));
    PyDict_SetItemString(py_options, "deallocate_error_fatal",
                         PyBool_FromLong(control->deallocate_error_fatal));
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
static PyObject* ssls_make_time_dict(const struct ssls_time_type *time){
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
// NB not static as it is used for nested control within SSLS Python interface
PyObject* ssls_make_inform_dict(const struct ssls_inform_type *inform){
    PyObject *py_inform = PyDict_New();

    PyDict_SetItemString(py_inform, "status",
                         PyLong_FromLong(inform->status));
    PyDict_SetItemString(py_inform, "alloc_status",
                         PyLong_FromLong(inform->alloc_status));
    PyDict_SetItemString(py_inform, "bad_alloc",
                         PyUnicode_FromString(inform->bad_alloc));
    PyDict_SetItemString(py_inform, "factorization_integer",
                         PyLong_FromLong(inform->factorization_integer));
    PyDict_SetItemString(py_inform, "factorization_real",
                         PyLong_FromLong(inform->factorization_real));
    PyDict_SetItemString(py_inform, "rank",
                         PyLong_FromLong(inform->rank));
    PyDict_SetItemString(py_inform, "rank_def",
                         PyBool_FromLong(inform->rank_def));
    PyDict_SetItemString(py_inform, "sls_inform",
                         sls_make_inform_dict(&inform->sls_inform));
    // Set time nested dictionary
    PyDict_SetItemString(py_inform, "time",
                         ssls_make_time_dict(&inform->time));

    return py_inform;
}

//  *-*-*-*-*-*-*-*-*-*-   SSLS_INITIALIZE    -*-*-*-*-*-*-*-*-*-*

static PyObject* py_ssls_initialize(PyObject *self){

    // Call ssls_initialize
    ssls_initialize(&data, &control, &status);

    // Record that SSLS has been initialised
    init_called = true;

    // Return options Python dictionary
    PyObject *py_options = ssls_make_options_dict(&control);
    return Py_BuildValue("N", py_options);
}

//  *-*-*-*-*-*-*-*-*-*-*-*-   SSLS_LOAD    -*-*-*-*-*-*-*-*-*-*-*-*

static PyObject* py_ssls_load(PyObject *self, PyObject *args, PyObject *keywds){
    PyArrayObject *py_H_row, *py_H_col, *py_H_ptr;
    PyArrayObject *tmp_H_row = NULL, *tmp_H_col = NULL, *tmp_H_ptr = NULL;
    PyArrayObject *py_A_row, *py_A_col, *py_A_ptr;
    PyArrayObject *tmp_A_row = NULL, *tmp_A_col = NULL, *tmp_A_ptr = NULL;
    PyArrayObject *py_C_row, *py_C_col, *py_C_ptr;
    PyArrayObject *tmp_C_row = NULL, *tmp_C_col = NULL, *tmp_C_ptr = NULL;
    PyObject *py_options = NULL;
    const ipc_ *H_row = NULL, *H_col = NULL, *H_ptr = NULL;
    const ipc_ *A_row = NULL, *A_col = NULL, *A_ptr = NULL;
    const ipc_ *C_row = NULL, *C_col = NULL, *C_ptr = NULL;
    const char *H_type, *A_type, *C_type;
    int n, m, H_ne, A_ne, C_ne;

    // Check that package has been initialised
    if(!check_init(init_called))
        return NULL;

    // Parse positional and keyword arguments
    static char *kwlist[] = {"n","m",
                             "H_type","H_ne","H_row","H_col","H_ptr",
                             "A_type","A_ne","A_row","A_col","A_ptr",
                             "C_type","C_ne","C_row","C_col","C_ptr",
                             "options",NULL};

    if(!PyArg_ParseTupleAndKeywords(args, keywds, "iisiOOOsiOOOsiOOO|O",
                                    kwlist, &n, &m,
                                    &H_type, &H_ne, &py_H_row,
                                    &py_H_col, &py_H_ptr,
                                    &A_type, &A_ne, &py_A_row,
                                    &py_A_col, &py_A_ptr,
                                    &C_type, &C_ne, &py_C_row,
                                    &py_C_col, &py_C_ptr,
                                    &py_options))
        return NULL;

    // Check that array inputs are of correct type, size, and shape
    if(!(
        check_array_int("H_row", py_H_row, H_ne) &&
        check_array_int("H_col", py_H_col, H_ne) &&
        check_array_int("H_ptr", py_H_ptr, n+1)  &&
        check_array_int("A_row", py_A_row, A_ne) &&
        check_array_int("A_col", py_A_col, A_ne) &&
        check_array_int("A_ptr", py_A_ptr, m+1)  &&
        check_array_int("C_row", py_C_row, C_ne) &&
        check_array_int("C_col", py_C_col, C_ne) &&
        check_array_int("C_ptr", py_C_ptr, m+1)
        ))
        return NULL;

    // Convert NumPy integer H_row array to ipc_
    if((PyObject *) py_H_row != Py_None){
       tmp_H_row = (PyArrayObject *) PyArray_FROM_OTF((PyObject *) py_H_row, NPY_IPC, NPY_ARRAY_IN_ARRAY | NPY_ARRAY_FORCECAST);
       if(tmp_H_row == NULL) goto conversion_error;
       H_row = (const ipc_ *) PyArray_DATA(tmp_H_row);
    }

    // Convert NumPy integer H_col array to ipc_
    if((PyObject *) py_H_col != Py_None){
       tmp_H_col = (PyArrayObject *) PyArray_FROM_OTF((PyObject *) py_H_col, NPY_IPC, NPY_ARRAY_IN_ARRAY | NPY_ARRAY_FORCECAST);
       if(tmp_H_col == NULL) goto conversion_error;
       H_col = (const ipc_ *) PyArray_DATA(tmp_H_col);
    }

    // Convert NumPy integer H_ptr array to ipc_
    if((PyObject *) py_H_ptr != Py_None){
       tmp_H_ptr = (PyArrayObject *) PyArray_FROM_OTF((PyObject *) py_H_ptr, NPY_IPC, NPY_ARRAY_IN_ARRAY | NPY_ARRAY_FORCECAST);
       if(tmp_H_ptr == NULL) goto conversion_error;
       H_ptr = (const ipc_ *) PyArray_DATA(tmp_H_ptr);
    }

    // Convert NumPy integer A_row array to ipc_
    if((PyObject *) py_A_row != Py_None){
       tmp_A_row = (PyArrayObject *) PyArray_FROM_OTF((PyObject *) py_A_row, NPY_IPC, NPY_ARRAY_IN_ARRAY | NPY_ARRAY_FORCECAST);
       if(tmp_A_row == NULL) goto conversion_error;
       A_row = (const ipc_ *) PyArray_DATA(tmp_A_row);
    }

    // Convert NumPy integer A_col array to ipc_
    if((PyObject *) py_A_col != Py_None){
       tmp_A_col = (PyArrayObject *) PyArray_FROM_OTF((PyObject *) py_A_col, NPY_IPC, NPY_ARRAY_IN_ARRAY | NPY_ARRAY_FORCECAST);
       if(tmp_A_col == NULL) goto conversion_error;
       A_col = (const ipc_ *) PyArray_DATA(tmp_A_col);
    }

    // Convert NumPy integer A_ptr array to ipc_
    if((PyObject *) py_A_ptr != Py_None){
       tmp_A_ptr = (PyArrayObject *) PyArray_FROM_OTF((PyObject *) py_A_ptr, NPY_IPC, NPY_ARRAY_IN_ARRAY | NPY_ARRAY_FORCECAST);
       if(tmp_A_ptr == NULL) goto conversion_error;
       A_ptr = (const ipc_ *) PyArray_DATA(tmp_A_ptr);
    }

    // Convert NumPy integer C_row array to ipc_
    if((PyObject *) py_C_row != Py_None){
       tmp_C_row = (PyArrayObject *) PyArray_FROM_OTF((PyObject *) py_C_row, NPY_IPC, NPY_ARRAY_IN_ARRAY | NPY_ARRAY_FORCECAST);
       if(tmp_C_row == NULL) goto conversion_error;
       C_row = (const ipc_ *) PyArray_DATA(tmp_C_row);
    }

    // Convert NumPy integer C_col array to ipc_
    if((PyObject *) py_C_col != Py_None){
       tmp_C_col = (PyArrayObject *) PyArray_FROM_OTF((PyObject *) py_C_col, NPY_IPC, NPY_ARRAY_IN_ARRAY | NPY_ARRAY_FORCECAST);
       if(tmp_C_col == NULL) goto conversion_error;
       C_col = (const ipc_ *) PyArray_DATA(tmp_C_col);
    }

    // Convert NumPy integer C_ptr array to ipc_
    if((PyObject *) py_C_ptr != Py_None){
       tmp_C_ptr = (PyArrayObject *) PyArray_FROM_OTF((PyObject *) py_C_ptr, NPY_IPC, NPY_ARRAY_IN_ARRAY | NPY_ARRAY_FORCECAST);
       if(tmp_C_ptr == NULL) goto conversion_error;
       C_ptr = (const ipc_ *) PyArray_DATA(tmp_C_ptr);
    }

    // Reset control options
    ssls_reset_control(&control, &data, &status);

    // Update SSLS control options
    if(!ssls_update_control(&control, py_options))
        return NULL;

    // Call ssls_analyse_matrix
    ssls_import(&control, &data, &status, n, m,
                H_type, H_ne, H_row, H_col, H_ptr,
                A_type, A_ne, A_row, A_col, A_ptr,
                C_type, C_ne, C_row, C_col, C_ptr);

    // Cleanup refcounts
    Py_XDECREF(tmp_H_row);
    Py_XDECREF(tmp_H_col);
    Py_XDECREF(tmp_H_ptr);
    Py_XDECREF(tmp_A_row);
    Py_XDECREF(tmp_A_col);
    Py_XDECREF(tmp_A_ptr);
    Py_XDECREF(tmp_C_row);
    Py_XDECREF(tmp_C_col);
    Py_XDECREF(tmp_C_ptr);

    // Raise any status errors
    if(!check_error_codes(status))
        return NULL;

    // Return None boilerplate
    Py_INCREF(Py_None);
    return Py_None;

    // Handle errors on array conversion
    conversion_error:
        Py_XDECREF(tmp_H_row);
        Py_XDECREF(tmp_H_col);
        Py_XDECREF(tmp_H_ptr);
        Py_XDECREF(tmp_A_row);
        Py_XDECREF(tmp_A_col);
        Py_XDECREF(tmp_A_ptr);
        Py_XDECREF(tmp_C_row);
        Py_XDECREF(tmp_C_col);
        Py_XDECREF(tmp_C_ptr);
        return NULL;
}

//  *-*-*-*-*-*-*-*-*-*-*-*-   SSLS_FACTORIZE_MATRIX    -*-*-*-*-*-*-*-*-*-*-*-*

static PyObject* py_ssls_factorize_matrix(PyObject *self, PyObject *args, PyObject *keywds){
    PyArrayObject *py_H_val, *py_A_val, *py_C_val;
    double *H_val, *A_val, *C_val;
    int H_ne, A_ne, C_ne;

    // Check that package has been initialised
    if(!check_init(init_called))
        return NULL;

    // Parse positional and keyword arguments
    static char *kwlist[] = {"H_ne","H_val","A_ne","A_val",
                             "C_ne","C_val",NULL};

    if(!PyArg_ParseTupleAndKeywords(args, keywds, "iOiOiO",
                                    kwlist, &H_ne, &py_H_val, &A_ne,
                                    &py_A_val, &C_ne, &py_C_val ))
        return NULL;

    // Check that array inputs are of correct type, size, and shape

    if(!(check_array_double("H_val", py_H_val, H_ne) &&
         check_array_double("A_val", py_A_val, A_ne) &&
         check_array_double("C_val", py_C_val, C_ne)))
        return NULL;

    // Get array data pointers
    H_val = (double *) PyArray_DATA(py_H_val);
    A_val = (double *) PyArray_DATA(py_A_val);
    C_val = (double *) PyArray_DATA(py_C_val);

    // Call ssls_factorize_matrix
    ssls_factorize_matrix(&data, &status, H_ne, H_val, A_ne, A_val,
                          C_ne, C_val);
    // Raise any status errors
    if(!check_error_codes(status))
        return NULL;

    // Return None boilerplate
    Py_INCREF(Py_None);
    return Py_None;
}

//  *-*-*-*-*-*-*-*-*-*-   SSLS_SOLVE_SYSTEM  -*-*-*-*-*-*-*-*

static PyObject* py_ssls_solve_system(PyObject *self, PyObject *args, PyObject *keywds){
    PyArrayObject *py_sol;
    double *sol;
    int n, m;

    // Check that package has been initialised
    if(!check_init(init_called))
        return NULL;

    // Parse positional arguments
    static char *kwlist[] = {"n", "m", "rhs", NULL};
    if(!PyArg_ParseTupleAndKeywords(args, keywds, "iiO", kwlist, &n, &m, &py_sol))
        return NULL;

    // Check that array inputs are of correct type, size, and shape
    if(!check_array_double("b", py_sol, n + m))
        return NULL;

    // Get array data pointers
    sol = (double *) PyArray_DATA(py_sol);

    // Call ssls_solve_direct
    ssls_solve_system(&data, &status, n, m, sol);

    // Propagate any errors with the callback function
    if(PyErr_Occurred())
        return NULL;

    // Raise any status errors
    if(!check_error_codes(status))
        return NULL;

    // Return x
    return Py_BuildValue("O", py_sol);
}

//  *-*-*-*-*-*-*-*-*-*-   SSLS_INFORMATION   -*-*-*-*-*-*-*-*

static PyObject* py_ssls_information(PyObject *self){

    // Check that package has been initialised
    if(!check_init(init_called))
        return NULL;

    // Call ssls_information
    ssls_information(&data, &inform, &status);

    // Return status and inform Python dictionary
    PyObject *py_inform = ssls_make_inform_dict(&inform);
    return Py_BuildValue("N", py_inform);
}

//  *-*-*-*-*-*-*-*-*-*-   SSLS_TERMINATE   -*-*-*-*-*-*-*-*-*-*

static PyObject* py_ssls_terminate(PyObject *self){

    // Check that package has been initialised
    if(!check_init(init_called))
        return NULL;

    // Call ssls_terminate
    ssls_terminate(&data, &control, &inform);

    // Record that SSLS must be reinitialised if called again
    init_called = false;

    // Return None boilerplate
    Py_INCREF(Py_None);
    return Py_None;
}

//  *-*-*-*-*-*-*-*-*-*-   INITIALIZE SSLS PYTHON MODULE    -*-*-*-*-*-*-*-*-*-*

/* ssls python module method table */
static PyMethodDef ssls_module_methods[] = {
    {"initialize", (PyCFunction) py_ssls_initialize, METH_NOARGS, NULL},
    {"load", (PyCFunction) py_ssls_load, METH_VARARGS | METH_KEYWORDS, NULL},
    {"factorize_matrix", (PyCFunction) py_ssls_factorize_matrix,
      METH_VARARGS | METH_KEYWORDS, NULL},
    {"solve_system", (PyCFunction) py_ssls_solve_system, METH_VARARGS | METH_KEYWORDS, NULL},
    {"information", (PyCFunction) py_ssls_information, METH_NOARGS, NULL},
    {"terminate", (PyCFunction) py_ssls_terminate, METH_NOARGS, NULL},
    {NULL, NULL, 0, NULL}  /* Sentinel */
};

/* ssls python module documentation */

PyDoc_STRVAR(ssls_module_doc,

"Given (possibly rectangular) A and symmetric H and C,\n"
"form and factorize the block, symmetric matrix\n"
"  K_H =  ( H  A^T ),\n"
"         ( A  - C }\n"
"and find solutions to the block linear system\n"
"   ( G  A^T ) ( x ) = ( a ).\n"
"   ( A  - C } ( y )   ( b )\n"
"Full advantage is taken of any zero coefficients in the matrices H,\n"
"A and C.\n"
"\n"
"See $GALAHAD/html/Python/ssls.html for argument lists, call order\n"
"and other details.\n"
"\n"
);

/* ssls python module definition */
static struct PyModuleDef module = {
   PyModuleDef_HEAD_INIT,
   "ssls",               /* name of module */
   ssls_module_doc,      /* module documentation, may be NULL */
   -1,                  /* size of per-interpreter state of the module,or -1
                           if the module keeps state in global variables */
   ssls_module_methods   /* module methods */
};

/* Python module initialization */
PyMODINIT_FUNC PyInit_ssls(void) { // must be same as module name above
    import_array();  // for NumPy arrays
    return PyModule_Create(&module);
}
