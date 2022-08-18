//* \file bgo_pyiface.c */

/*
 * THIS VERSION: GALAHAD 4.1 - 2022-08-17 AT 14:30 GMT.
 *
 *-*-*-*-*-*-*-*-*-  GALAHAD_BGO PYTHON INTERFACE  *-*-*-*-*-*-*-*-*-*-
 *
 *  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
 *  Principal author: Jaroslav Fowkes & Nick Gould
 *
 *  History -
 *   originally released GALAHAD Version 4.1. August 17th 2022
 *
 *  For full documentation, see
 *   http://galahad.rl.ac.uk/galahad-www/specs.html
 */

#include "galahad_python.h"
#include "galahad_bgo.h"

/* Nested UGO control and inform prototypes */
bool ugo_update_control(struct ugo_control_type *control, PyObject *py_options);
PyObject* ugo_make_inform_dict(const struct ugo_inform_type *inform);

/* Module global variables */
static void *data;                       // private internal data
static struct bgo_control_type control;  // control struct
static struct bgo_inform_type inform;    // inform struct
static bool init_called = false;         // record if initialise was called
static int status = 0;                   // exit status

//  *-*-*-*-*-*-*-*-*-*-   CALLBACK FUNCTIONS    -*-*-*-*-*-*-*-*-*-*

/* Python eval_* function pointers */
static PyObject *py_eval_f = NULL;
static PyObject *py_eval_g = NULL;

/* C eval_* function wrappers */
static int eval_f(int n, const double x[], double *f, const void *userdata){

    // Wrap input array as NumPy array
    npy_intp xdim[] = {n};
    PyObject *py_x = PyArray_SimpleNewFromData(1, xdim, NPY_DOUBLE, (void *) x);

    // Build Python argument list
    PyObject *arglist = Py_BuildValue("(O)", py_x);

    // Call Python eval_f
    PyObject *result = PyObject_CallObject(py_eval_f, arglist);
    Py_DECREF(py_x);    // Free py_x memory
    Py_DECREF(arglist); // Free arglist memory

    // Check that eval was successful
    if(!result)
        return -1;

    // Extract single double return value
    if(!parse_double("eval_f return value", result, f)){
        Py_DECREF(result); // Free result memory
        return -1;
    }

    // Free result memory
    Py_DECREF(result);

    return 0;
}

static int eval_g(int n, const double x[], double g[], const void *userdata){

    // Wrap input array as NumPy array
    npy_intp xdim[] = {n};
    PyArrayObject *py_x = (PyArrayObject*)
       PyArray_SimpleNewFromData(1, xdim, NPY_DOUBLE, (void *) x);

    // Build Python argument list
    PyObject *arglist = Py_BuildValue("(O)", py_x);

    // Call Python eval_g
    PyObject *result = PyObject_CallObject(py_eval_g, arglist);
    Py_DECREF(py_x);    // Free py_x memory
    Py_DECREF(arglist); // Free arglist memory

    // Check that eval was successful
    if(!result)
        return -1;

    // Check return value is of correct type, size, and shape
    if(!check_array_double("eval_g return value", (PyArrayObject*) result, n)){
        Py_DECREF(result); // Free result memory
        return -1;
    }

    // Get return value data pointer and copy data into g
    const double *gval = (double *) PyArray_DATA((PyArrayObject*) result);
    for(int i=0; i<n; i++) {
        g[i] = gval[i];
    }

    // Free result memory
    Py_DECREF(result);

    return 0;
}

//  *-*-*-*-*-*-*-*-*-*-   UPDATE CONTROL    -*-*-*-*-*-*-*-*-*-*

/* Update the control options: use C defaults but update any passed via Python*/
static bool bgo_update_control(struct bgo_control_type *control,
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

        // Parse each int option
        if(strcmp(key_name, "print_level") == 0){
            if(!parse_int_option(value, "print_level", &control->print_level))
                return false;
            continue;
        }
        // ... other int options ...

        // Parse each float/double option
        if(strcmp(key_name, "obj_unbounded") == 0){
            if(!parse_double_option(value, "obj_unbounded",
                                    &control->obj_unbounded))
                return false;
            continue;
        }
        // ... other float/double options ...

        // Parse each bool option
        if(strcmp(key_name, "space_critical") == 0){
            if(!parse_bool_option(value, "space_critical",
                                  &control->space_critical))
                return false;
            continue;
        }
        // ... other bool options ...

        // Parse each char option
        if(strcmp(key_name, "prefix") == 0){
            if(!parse_char_option(value, "prefix",
                                  control->prefix))
                return false;
            continue;
        }
        // ... other char options ...

        // Parse nested control
        if(strcmp(key_name, "ugo_options") == 0){
            if(!ugo_update_control(&control->ugo_control, value))
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

//  *-*-*-*-*-*-*-*-*-*-   MAKE TIME    -*-*-*-*-*-*-*-*-*-*

/* Take the time struct from C and turn it into a python dictionary */
static PyObject* bgo_make_time_dict(const struct bgo_time_type *time){
    PyObject *py_time = PyDict_New();

    // Set float/double time entries
    PyDict_SetItemString(py_time, "total", PyFloat_FromDouble(time->total));
    PyDict_SetItemString(py_time, "univariate_global",
           PyFloat_FromDouble(time->univariate_global));
    PyDict_SetItemString(py_time, "multivariate_local",
           PyFloat_FromDouble(time->multivariate_local));
    PyDict_SetItemString(py_time, "clock_total",
           PyFloat_FromDouble(time->clock_total));
    PyDict_SetItemString(py_time, "clock_univariate_global",
           PyFloat_FromDouble(time->clock_univariate_global));
    PyDict_SetItemString(py_time, "clock_multivariate_local",
           PyFloat_FromDouble(time->clock_multivariate_local));

    return py_time;
}

//  *-*-*-*-*-*-*-*-*-*-   MAKE INFORM    -*-*-*-*-*-*-*-*-*-*

/* Take the inform struct from C and turn it into a python dictionary */
static PyObject* bgo_make_inform_dict(const struct bgo_inform_type *inform){
    PyObject *py_inform = PyDict_New();

    // Set int inform entries
    PyDict_SetItemString(py_inform, "f_eval", PyLong_FromLong(inform->f_eval));
    // ... other int inform entries ...

    // Set float/double inform entries
    PyDict_SetItemString(py_inform, "obj", PyFloat_FromDouble(inform->obj));
    // ... other float/double inform entries ...

    // Set bool inform entries
    //PyDict_SetItemString(py_inform, "used_grad", PyBool_FromLong(inform->used_grad));
    // ... other bool inform entries ...

    // Set char inform entries
    //PyDict_SetItemString(py_inform, "name", PyUnicode_FromString(inform->name));
    // ... other char inform entries ...

    // Set time nested dictionary
    PyDict_SetItemString(py_inform, "time", bgo_make_time_dict(&inform->time));

    // Set UGO nested dictionary
    PyDict_SetItemString(py_inform, "ugo_inform",
                         ugo_make_inform_dict(&inform->ugo_inform));

    return py_inform;
}

//  *-*-*-*-*-*-*-*-*-*-   BGO_INITIALIZE    -*-*-*-*-*-*-*-*-*-*

PyDoc_STRVAR(py_bgo_initialize_doc,
"bgo.initialize()\n"
"\n"
"Set default option values and initialize private data\n"
"\n"
);

static PyObject* py_bgo_initialize(PyObject *self){

    // Call bgo_initialize
    bgo_initialize(&data, &control, &status);

    // Record that BGO has been initialised
    init_called = true;

    // Return None boilerplate
    Py_INCREF(Py_None);
    return Py_None;
}

//  *-*-*-*-*-*-*-*-*-*-*-*-   BGO_LOAD    -*-*-*-*-*-*-*-*-*-*-*-*
//  NB import is a python reserved keyword so changed to load here

PyDoc_STRVAR(py_bgo_load_doc,
"bgo.load(n, x_l, x_u, H_type, ne, H_row, H_col, H_ptr, options=None)\n"
"\n"
"Import problem data into internal storage prior to solution.\n"
"\n"
"Parameters\n"
"----------\n"
"n : int\n"
"    holds the number of variables.\n"
"x_l : ndarray(n)\n"
"    holds the values :math:`x^l` of the lower bounds on the optimization variables :math:`x`.\n"
"x_u : ndarray(n)\n"
"    holds the values :math:`x^u` of the upper bounds on the optimization variables :math:`x`.\n"
"H_type : string\n"
"    specifies the symmetric storage scheme used for the Hessian. It should\n"
"    be one of 'coordinate', 'sparse_by_rows', 'dense','diagonal' or 'absent',\n"
"    the latter if access to the Hessian is via matrix-vector products; lower\n"
"    or upper case variants are allowed.\n"
"ne : int\n"
"    holds the number of entries in the  lower triangular part of :math:`H` in the\n"
"    sparse co-ordinate storage scheme. It need not be set for any of the\n"
"    other three schemes.\n"
"H_row : ndarray(ne)\n"
"    holds the row indices of the lower triangular part of :math:`H` in the\n"
"    sparse co-ordinate storage scheme. It need not be set for any of the\n"
"    other three schemes, and in this case can be None\n"
"H_col : ndarray(ne)\n"
"    holds the column indices of the  lower triangular part of :math:`H` in either\n"
"    the sparse co-ordinate, or the sparse row-wise  storage scheme.\n"
"    It need not be set when the dense or diagonal storage schemes are used,\n"
"    and in this case can be None\n"
"H_ptr : ndarray(n+1)\n"
"    holds the starting position of each row of the lower triangular part of :math:`H`,\n"
"    as well as the total number of entries plus one, in the sparse row-wise\n"
"    storage scheme. It need not be set when the other schemes are used,\n"
"    and in this case can be None\n"
"options : dict, optional\n"
"    dictionary of control options\n"
"\n"
);

static PyObject* py_bgo_load(PyObject *self, PyObject *args, PyObject *keywds){
    PyArrayObject *py_x_l, *py_x_u, *py_H_row, *py_H_col, *py_H_ptr;
    PyObject *py_options = NULL;
    double *x_l, *x_u;
    int *H_row, *H_col, *H_ptr;
    const char *H_type;
    int n, ne;

    // Check that package has been initialised
    if(!check_init(init_called))
        return NULL;

    // Parse positional and keyword arguments
    static char *kwlist[] = {"n","x_l","x_u","H_type","ne",
                             "H_row","H_col","H_ptr","options",NULL};
    if(!PyArg_ParseTupleAndKeywords(args, keywds, "iOOsiOOO|O", kwlist, &n,
                                    &py_x_l, &py_x_u, &H_type, &ne, &py_H_row,
                                    &py_H_col, &py_H_ptr, &py_options))
        return NULL;

    // Check that array inputs are of correct type, size, and shape
//    if((
    if(!(
        check_array_double("x_l", py_x_l, n) &&
        check_array_double("x_u", py_x_u, n) &&
        check_array_int("H_row", py_H_row, ne) &&
        check_array_int("H_col", py_H_col, ne) &&
        check_array_int("H_ptr", py_H_ptr, n+1)
        ))
        return NULL;

    // Get array data pointers
    x_l = (double *) PyArray_DATA(py_x_l);
        printf("x_l: ");
        for(int i = 0; i < n; i++) printf("%f ", x_l[i]);
        printf("\n");
    x_u = (double *) PyArray_DATA(py_x_u);
        printf("x_u: ");
        for(int i = 0; i < n; i++) printf("%f ", x_u[i]);
        printf("\n");
    if((PyObject *) py_H_row == Py_None){
        printf("row: null\n");
      H_row = NULL;
     }else{
        printf("row: set\n");
      H_row = (int *) PyArray_DATA(py_H_row);
    }
    if((PyObject *) py_H_col == Py_None){
        printf("col: null\n");
      H_col = NULL;
    }else{
        printf("col: set\n");
//      H_col = (int *) PyArray_DATA(py_H_col);

        long int * H_col_long = (long int *) PyArray_DATA(py_H_col);

        printf("long col: ");
        for(int i = 0; i < ne; i++) printf("%li ", H_col_long[i]);
        printf("\n");


        int j;
        int H_col[ne];
        for(int i = 0; i < ne; i++) {
          j = (int) H_col_long[i];
          printf("%d ", j);
          H_col[i] = j;
        }
        printf("converted \n");

        printf("col: ");
          for(int i = 0; i < ne; i++) printf("%d ", H_col[i]);
        printf("\n");
    }

    if((PyObject *) py_H_ptr == Py_None){
        printf("ptr: null\n");
      H_ptr = NULL;
    }else{
        printf("ptr: set\n");
//      H_ptr = (int *) PyArray_DATA(py_H_ptr);
        long int * H_ptr_long = (long int *) PyArray_DATA(py_H_ptr);
//        H_ptr = H_ptr_long;

        printf("long ptr: ");
        for(int i = 0; i < n+1; i++) printf("%li ", H_ptr_long[i]);
        printf("\n");


        int j;
        int H_ptr[n+1];
        for(int i = 0; i < n+1; i++) {
          j = (int) H_ptr_long[i];
          printf("%d ", j);
          H_ptr[i] = j;
        }
        printf("converted \n");

        printf("ptr: ");
          for(int i = 0; i < n+1; i++) printf("%d ", H_ptr[i]);
        printf("\n");
    }

    // Reset control options
    bgo_reset_control(&control, &data, &status);

    // Update BGO control options
    if(!bgo_update_control(&control, py_options))
       return NULL;

    printf("import in\n");
    // Call bgo_import
    bgo_import(&control, &data, &status, n, x_l, x_u, H_type, ne,
               H_row, H_col, H_ptr);
    printf("import out \n");

    // Raise any status errors
    if(!check_error_codes(status))
        return NULL;

    // Return None boilerplate
    Py_INCREF(Py_None);
    printf("out \n");
    return Py_None;
}

//  *-*-*-*-*-*-*-*-*-*-   BGO_SOLVE   -*-*-*-*-*-*-*-*

PyDoc_STRVAR(py_bgo_solve_doc,
"x, g = bgo.solve(n, x, g, eval_f, eval_g)\n"
"\n"
"Find an approximation to the global minimizer of a given function subject to\n"
" simple bounds on the variables using a multistart trust-region method.\n"
"\n"
"Parameters\n"
"----------\n"
"n : int\n"
"    holds the number of variables.\n"
"x : ndarray(n)\n"
"    holds the values of optimization variables :math:`x`.\n"
"g : ndarray(n)\n"
"    holds the gradient :math:`\\nabla f(x)` of the objective function.\n"
"eval_f : callable\n"
"    a user-defined function that must have the signature:\n"
"\n"
"     ``f = eval_f(x)``\n"
"\n"
"    The value of the objective function :math:`f(x)`\n"
"    evaluated at :math:`x` must be assigned to ``f``.\n"
"eval_g : callable\n"
"    a user-defined function that must have the signature:\n"
"\n"
"     ``g = eval_g(x)``\n"
"\n"
"    The components of the gradient :math:`\\nabla f(x)` of the\n"
"    objective function evaluated at :math:`x` must be assigned to ``g``.\n"
"\n"
"Returns\n"
"-------\n"
"x : ndarray(n)\n"
"    holds the value of the approximate global minimizer :math:`x` after a\n"
"    successful call.\n"
"g : ndarray(n)\n"
"    holds the value of the gradient of the objective function :math:`\\nabla f(x)`\n"
"    at the approximate global minimizer :math:`x` after a successful call.\n"
"\n"
);

static PyObject* py_bgo_solve(PyObject *self, PyObject *args){
    PyArrayObject *py_x, *py_g;
    PyObject *temp_f, *temp_g;
    double *x, *g;
    int n;

    // Check that package has been initialised
    if(!check_init(init_called))
        return NULL;

    // Parse positional arguments
    if(!PyArg_ParseTuple(args, "iOOOO", &n, &py_x, &py_g, &temp_f, &temp_g))
        return NULL;

    // Check that array inputa are of correct type, size, and shape
    if(!(
        check_array_double("x", py_x, n) &&
        check_array_double("g", py_g, n)
        ))
        return NULL;

    // Get array data pointers
    x = (double *) PyArray_DATA(py_x);
    g = (double *) PyArray_DATA(py_g);

    // Check that functions are callable
    if(!(
        check_callable(temp_f) &&
        check_callable(temp_g)
        ))
        return NULL;

    // Store functions
    Py_XINCREF(temp_f);         /* Add a reference to new callback */
    Py_XDECREF(py_eval_f);      /* Dispose of previous callback */
    py_eval_f = temp_f;         /* Remember new callback */
    Py_XINCREF(temp_g);         /* Add a reference to new callback */
    Py_XDECREF(py_eval_g);      /* Dispose of previous callback */
    py_eval_g = temp_g;         /* Remember new callback */

    // Call bgo_solve_direct
    status = 1; // set status to 1 on entry
    bgo_solve_with_mat(&data, NULL, &status, n, x, g, -1, eval_f, eval_g,
                       NULL, NULL, NULL);

    // Propagate any errors with the callback function
    if(PyErr_Occurred())
        return NULL;

    // Raise any status errors
    if(!check_error_codes(status))
        return NULL;

    // Return status and x
    return Py_BuildValue("OO", py_x, py_g);
}

//  *-*-*-*-*-*-*-*-*-*-   BGO_INFORMATION   -*-*-*-*-*-*-*-*

PyDoc_STRVAR(py_bgo_information_doc,
"inform = bgo.information()\n"
"\n"
"Provide output information\n"
"\n"
"Returns\n"
"-------\n"
"inform : dict\n"
"    dictionary containing output information\n"
"\n"
);

static PyObject* py_bgo_information(PyObject *self){

    // Check that package has been initialised
    if(!check_init(init_called))
        return NULL;

    // Call bgo_information
    bgo_information(&data, &inform, &status);

    // Return status and inform Python dictionary
    PyObject *py_inform = bgo_make_inform_dict(&inform);
    return Py_BuildValue("O", py_inform);
}

//  *-*-*-*-*-*-*-*-*-*-   BGO_TERMINATE   -*-*-*-*-*-*-*-*-*-*

PyDoc_STRVAR(py_bgo_terminate_doc,
"bgo.terminate()\n"
"\n"
"Deallocate all internal private storage\n"
"\n"
);

static PyObject* py_bgo_terminate(PyObject *self){

    // Check that package has been initialised
    if(!check_init(init_called))
        return NULL;

    // Call bgo_terminate
    bgo_terminate(&data, &control, &inform);

    // Return None boilerplate
    Py_INCREF(Py_None);
    return Py_None;
}

//  *-*-*-*-*-*-*-*-*-*-   INITIALIZE BGO PYTHON MODULE    -*-*-*-*-*-*-*-*-*-*

/* bgo python module method table */
static PyMethodDef bgo_module_methods[] = {
    {"initialize", (PyCFunction) py_bgo_initialize, METH_NOARGS,
      py_bgo_initialize_doc},
    {"load", (PyCFunction) py_bgo_load, METH_VARARGS | METH_KEYWORDS,
      py_bgo_load_doc},
    {"solve", (PyCFunction) py_bgo_solve, METH_VARARGS,
      py_bgo_solve_doc},
    {"information", (PyCFunction) py_bgo_information, METH_NOARGS,
      py_bgo_information_doc},
    {"terminate", (PyCFunction) py_bgo_terminate, METH_NOARGS,
      py_bgo_terminate_doc},
    {NULL, NULL, 0, NULL}  /* Sentinel */
};

/* bgo python module documentation */
PyDoc_STRVAR(bgo_module_doc,
"The bgo package uses a multi-start trust-region method to find an\n"
"  approximation to the global minimizer of a differentiable objective\n"
"  function :math:`f(x)` of n variables :math:`x`, subject to simple bounds\n"
"  :math:`x^l <= x <= x^u` on the variables.\n"
"  Here, any of the components of the vectors of bounds :math:`x^l` and\n"
"  :math:`x^u` may be infinite. The method offers the choice of direct\n"
"  and iterative solution of the key trust-region subproblems, and\n"
"  is suitable for large problems. First derivatives are required,\n"
"  and if second derivatives can be calculated, they will be exploited---if\n"
"  the product of second derivatives with a vector may be found but\n"
"  not the derivatives themselves, that may also be exploited.\n"
);

/* bgo python module definition */
static struct PyModuleDef module = {
   PyModuleDef_HEAD_INIT,
   "bgo",               /* name of module */
   bgo_module_doc,      /* module documentation, may be NULL */
   -1,                  /* size of per-interpreter state of the module,or -1
                           if the module keeps state in global variables */
   bgo_module_methods   /* module methods */
};

/* Python module initialization */
PyMODINIT_FUNC PyInit_bgo(void) { // must be same as module name above
    import_array();  // for NumPy arrays
    return PyModule_Create(&module);
}
