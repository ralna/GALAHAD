//* \file rpd_pyiface.c */

/*
 * THIS VERSION: GALAHAD 4.1 - 2023-05-12 AT 08:40 GMT.
 *
 *-*-*-*-*-*-*-*-*-  GALAHAD_RPD PYTHON INTERFACE  *-*-*-*-*-*-*-*-*-*-
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

#include <string.h>
#include "galahad_python.h"
#include "galahad_rpd.h"

/* Module global variables */
static void *data;                       // private internal data
static struct rpd_control_type control;  // control struct
static struct rpd_inform_type inform;    // inform struct
static bool init_called = false;         // record if initialise was called
static int status = 0;                   // exit status

//  *-*-*-*-*-*-*-*-*-*-   UPDATE CONTROL    -*-*-*-*-*-*-*-*-*-*

/* Update the control options: use C defaults but update any passed via Python*/
// NB not static as it is used for nested control within QP Python interfaces
bool rpd_update_control(struct rpd_control_type *control,
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
        if(strcmp(key_name, "qplib") == 0){
            if(!parse_int_option(value, "qplib",
                                  &control->qplib))
                return false;
            continue;
        }
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
        // Otherwise unrecognised option
        PyErr_Format(PyExc_ValueError,
          "unrecognised option options['%s']\n", key_name);
        return false;
    }

    return true; // success
}

//  *-*-*-*-*-*-*-*-*-*-   MAKE OPTIONS    -*-*-*-*-*-*-*-*-*-*

/* Take the control struct from C and turn it into a python options dict */
// NB not static as it is used for nested inform within QP Python interfaces
PyObject* rpd_make_options_dict(const struct rpd_control_type *control){
    PyObject *py_options = PyDict_New();

    PyDict_SetItemString(py_options, "qplib",
                         PyLong_FromLong(control->qplib));
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
    return py_options;
}

//  *-*-*-*-*-*-*-*-*-*-   MAKE INFORM    -*-*-*-*-*-*-*-*-*-*

/* Take the inform struct from C and turn it into a python dictionary */
// NB not static as it is used for nested control within QP Python interfaces
PyObject* rpd_make_inform_dict(const struct rpd_inform_type *inform){
    PyObject *py_inform = PyDict_New();

    PyDict_SetItemString(py_inform, "status",
                         PyLong_FromLong(inform->status));
    PyDict_SetItemString(py_inform, "alloc_status",
                         PyLong_FromLong(inform->alloc_status));
    PyDict_SetItemString(py_inform, "io_status",
                         PyLong_FromLong(inform->io_status));
    PyDict_SetItemString(py_inform, "line",
                         PyLong_FromLong(inform->line));
    PyDict_SetItemString(py_inform, "p_type",
                         PyUnicode_FromString(inform->p_type));
    PyDict_SetItemString(py_inform, "bad_alloc",
                         PyUnicode_FromString(inform->bad_alloc));
    return py_inform;
}

//  *-*-*-*-*-*-*-*-*-*-   RPD_INITIALIZE    -*-*-*-*-*-*-*-*-*-*

static PyObject* py_rpd_initialize(PyObject *self){

    // Call rpd_initialize
    rpd_initialize(&data, &control, &status);

    // Record that RPD has been initialised
    init_called = true;

    // Return options Python dictionary
    PyObject *py_options = rpd_make_options_dict(&control);
    return Py_BuildValue("N", py_options);
}

//  *-*-*-*-*-*-*-*-*-*-*-*-*-*-   RPD_GET_STATS   -*-*-*-*-*-*-*-*-*-*-*-*-*

static PyObject* py_rpd_get_stats(PyObject *self, PyObject *args, PyObject *keywds){
    PyObject *py_options = NULL;
    const char *qplib_file;
    char p_type[4] = "";
    ipc_ n, m, h_ne, a_ne, h_c_ne;

    // Check that package has been initialised
    if(!check_init(init_called))
        return NULL;

    // Parse positional and keyword arguments
    static char *kwlist[] = {"qplib_file","options",NULL};
    if(!PyArg_ParseTupleAndKeywords(args, keywds, "s|O", kwlist,
                                    &qplib_file, &py_options))
        return NULL;

    // Update RPD control options
    if(!rpd_update_control(&control, py_options))
        return NULL;

    // Call rpd_get_stats (which does not alter the file name)
    rpd_get_stats((char *) qplib_file, (ipc_) strlen(qplib_file), &control,
                  &data, &status, p_type, &n, &m, &h_ne, &a_ne, &h_c_ne);

    // Raise any status errors
    if(!check_error_codes(status))
        return NULL;

    // Return the problem type and dimensions
    p_type[3] = '\0';
    return Py_BuildValue("siiiii", p_type, (int) n, (int) m, (int) h_ne,
                         (int) a_ne, (int) h_c_ne);
}

//  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-   RPD_GET_F   -*-*-*-*-*-*-*-*-*-*-*-*-*-*

static PyObject* py_rpd_get_f(PyObject *self){
    double f;

    // Check that package has been initialised
    if(!check_init(init_called))
        return NULL;

    // Call rpd_get_f
    rpd_get_f(&data, &status, &f);

    // Raise any status errors
    if(!check_error_codes(status))
        return NULL;

    // Return f
    return Py_BuildValue("d", f);
}

// Get a real vector of length len (n or m) with get_vector, and return it

static PyObject* rpd_vector(PyObject *args, PyObject *keywds, char *name,
                            void (*get_vector)(void **, ipc_ *, ipc_,
                                               double *)){
    int len;

    // Check that package has been initialised
    if(!check_init(init_called))
        return NULL;

    // Parse positional and keyword arguments
    static char *kwlist_n[] = {"n",NULL};
    static char *kwlist_m[] = {"m",NULL};
    if(!PyArg_ParseTupleAndKeywords(args, keywds, "i",
                                    name[0] == 'n' ? kwlist_n : kwlist_m,
                                    &len))
        return NULL;

    // Create the NumPy array and fill it
    npy_intp dim[] = {len};
    PyArrayObject *py_v =
      (PyArrayObject *) PyArray_SimpleNew(1, dim, NPY_DOUBLE);
    get_vector(&data, &status, len, (double *) PyArray_DATA(py_v));

    // Raise any status errors
    if(!check_error_codes(status)){
        Py_DECREF(py_v);
        return NULL;
    }
    return Py_BuildValue("N", py_v);
}

//  *-*-*-*-*-*-*-*-*-*-*-*-   RPD_GET_G, X, Y, Z   -*-*-*-*-*-*-*-*-*-*-*-*

static PyObject* py_rpd_get_g(PyObject *self, PyObject *args, PyObject *keywds){
    return rpd_vector(args, keywds, "n", rpd_get_g);
}

static PyObject* py_rpd_get_x(PyObject *self, PyObject *args, PyObject *keywds){
    return rpd_vector(args, keywds, "n", rpd_get_x);
}

static PyObject* py_rpd_get_y(PyObject *self, PyObject *args, PyObject *keywds){
    return rpd_vector(args, keywds, "m", rpd_get_y);
}

static PyObject* py_rpd_get_z(PyObject *self, PyObject *args, PyObject *keywds){
    return rpd_vector(args, keywds, "n", rpd_get_z);
}

// Get lower and upper bounds of length len (n or m) with get_bounds, and
// return them

static PyObject* rpd_bounds(PyObject *args, PyObject *keywds, char *name,
                            void (*get_bounds)(void **, ipc_ *, ipc_,
                                               double *, double *)){
    int len;

    // Check that package has been initialised
    if(!check_init(init_called))
        return NULL;

    // Parse positional and keyword arguments
    static char *kwlist_n[] = {"n",NULL};
    static char *kwlist_m[] = {"m",NULL};
    if(!PyArg_ParseTupleAndKeywords(args, keywds, "i",
                                    name[0] == 'n' ? kwlist_n : kwlist_m,
                                    &len))
        return NULL;

    // Create the NumPy arrays and fill them
    npy_intp dim[] = {len};
    PyArrayObject *py_l =
      (PyArrayObject *) PyArray_SimpleNew(1, dim, NPY_DOUBLE);
    PyArrayObject *py_u =
      (PyArrayObject *) PyArray_SimpleNew(1, dim, NPY_DOUBLE);
    get_bounds(&data, &status, len, (double *) PyArray_DATA(py_l),
               (double *) PyArray_DATA(py_u));

    // Raise any status errors
    if(!check_error_codes(status)){
        Py_DECREF(py_l);
        Py_DECREF(py_u);
        return NULL;
    }
    return Py_BuildValue("NN", py_l, py_u);
}

//  *-*-*-*-*-*-*-*-*-*-*-*-*-*-   RPD_GET_XLU, CLU   -*-*-*-*-*-*-*-*-*-*-*-*

static PyObject* py_rpd_get_xlu(PyObject *self, PyObject *args, PyObject *keywds){
    return rpd_bounds(args, keywds, "n", rpd_get_xlu);
}

static PyObject* py_rpd_get_clu(PyObject *self, PyObject *args, PyObject *keywds){
    return rpd_bounds(args, keywds, "m", rpd_get_clu);
}

// Get a matrix with ne entries in co-ordinate form with get_matrix, and
// return its row and column indices and values

static PyObject* rpd_matrix(PyObject *args, PyObject *keywds,
                            void (*get_matrix)(void **, ipc_ *, ipc_,
                                               ipc_ *, ipc_ *, double *)){
    int ne;

    // Check that package has been initialised
    if(!check_init(init_called))
        return NULL;

    // Parse positional and keyword arguments
    static char *kwlist[] = {"ne",NULL};
    if(!PyArg_ParseTupleAndKeywords(args, keywds, "i", kwlist, &ne))
        return NULL;

    // Create the NumPy arrays and fill them
    npy_intp dim[] = {ne};
    PyArrayObject *py_row =
      (PyArrayObject *) PyArray_ZEROS(1, dim, NPY_IPC, 0);
    PyArrayObject *py_col =
      (PyArrayObject *) PyArray_ZEROS(1, dim, NPY_IPC, 0);
    PyArrayObject *py_val =
      (PyArrayObject *) PyArray_SimpleNew(1, dim, NPY_DOUBLE);
    get_matrix(&data, &status, ne, (ipc_ *) PyArray_DATA(py_row),
               (ipc_ *) PyArray_DATA(py_col), (double *) PyArray_DATA(py_val));

    // Raise any status errors
    if(!check_error_codes(status)){
        Py_DECREF(py_row);
        Py_DECREF(py_col);
        Py_DECREF(py_val);
        return NULL;
    }
    return Py_BuildValue("NNN", py_row, py_col, py_val);
}

//  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-   RPD_GET_H, A   -*-*-*-*-*-*-*-*-*-*-*-*-*

static PyObject* py_rpd_get_h(PyObject *self, PyObject *args, PyObject *keywds){
    return rpd_matrix(args, keywds, rpd_get_h);
}

static PyObject* py_rpd_get_a(PyObject *self, PyObject *args, PyObject *keywds){
    return rpd_matrix(args, keywds, rpd_get_a);
}

//  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-   RPD_GET_H_C   -*-*-*-*-*-*-*-*-*-*-*-*-*

static PyObject* py_rpd_get_h_c(PyObject *self, PyObject *args, PyObject *keywds){
    int ne;

    // Check that package has been initialised
    if(!check_init(init_called))
        return NULL;

    // Parse positional and keyword arguments
    static char *kwlist[] = {"ne",NULL};
    if(!PyArg_ParseTupleAndKeywords(args, keywds, "i", kwlist, &ne))
        return NULL;

    // Create the NumPy arrays and fill them
    npy_intp dim[] = {ne};
    PyArrayObject *py_ptr =
      (PyArrayObject *) PyArray_ZEROS(1, dim, NPY_IPC, 0);
    PyArrayObject *py_row =
      (PyArrayObject *) PyArray_ZEROS(1, dim, NPY_IPC, 0);
    PyArrayObject *py_col =
      (PyArrayObject *) PyArray_ZEROS(1, dim, NPY_IPC, 0);
    PyArrayObject *py_val =
      (PyArrayObject *) PyArray_SimpleNew(1, dim, NPY_DOUBLE);
    rpd_get_h_c(&data, &status, ne, (ipc_ *) PyArray_DATA(py_ptr),
                (ipc_ *) PyArray_DATA(py_row), (ipc_ *) PyArray_DATA(py_col),
                (double *) PyArray_DATA(py_val));

    // Raise any status errors
    if(!check_error_codes(status)){
        Py_DECREF(py_ptr);
        Py_DECREF(py_row);
        Py_DECREF(py_col);
        Py_DECREF(py_val);
        return NULL;
    }
    return Py_BuildValue("NNNN", py_ptr, py_row, py_col, py_val);
}

//  *-*-*-*-*-*-*-*-*-*-*-*-*-*-   RPD_GET_X_TYPE   -*-*-*-*-*-*-*-*-*-*-*-*-*

static PyObject* py_rpd_get_x_type(PyObject *self, PyObject *args, PyObject *keywds){
    int n;

    // Check that package has been initialised
    if(!check_init(init_called))
        return NULL;

    // Parse positional and keyword arguments
    static char *kwlist[] = {"n",NULL};
    if(!PyArg_ParseTupleAndKeywords(args, keywds, "i", kwlist, &n))
        return NULL;

    // Create the NumPy array and fill it
    npy_intp dim[] = {n};
    PyArrayObject *py_x_type =
      (PyArrayObject *) PyArray_ZEROS(1, dim, NPY_IPC, 0);
    rpd_get_x_type(&data, &status, n, (ipc_ *) PyArray_DATA(py_x_type));

    // Raise any status errors
    if(!check_error_codes(status)){
        Py_DECREF(py_x_type);
        return NULL;
    }
    return Py_BuildValue("N", py_x_type);
}

//  *-*-*-*-*-*-*-*-*-*-   RPD_INFORMATION   -*-*-*-*-*-*-*-*

static PyObject* py_rpd_information(PyObject *self){

    // Check that package has been initialised
    if(!check_init(init_called))
        return NULL;

    // Call rpd_information
    rpd_information(&data, &inform, &status);

    // Return status and inform Python dictionary
    PyObject *py_inform = rpd_make_inform_dict(&inform);
    return Py_BuildValue("N", py_inform);
}

//  *-*-*-*-*-*-*-*-*-*-   RPD_TERMINATE   -*-*-*-*-*-*-*-*-*-*

static PyObject* py_rpd_terminate(PyObject *self){

    // Check that package has been initialised
    if(!check_init(init_called))
        return NULL;

    // Call rpd_terminate
    rpd_terminate(&data, &control, &inform);

    // require future calls start with rpd_initialize
    init_called = false;

    // Return None boilerplate
    Py_INCREF(Py_None);
    return Py_None;
}

//  *-*-*-*-*-*-*-*-*-*-   INITIALIZE RPD PYTHON MODULE    -*-*-*-*-*-*-*-*-*-*

/* rpd python module method table */
static PyMethodDef rpd_module_methods[] = {
    {"initialize", (PyCFunction) py_rpd_initialize, METH_NOARGS,NULL},
    {"get_stats", (PyCFunction) py_rpd_get_stats, METH_VARARGS | METH_KEYWORDS, NULL},
    {"get_g", (PyCFunction) py_rpd_get_g, METH_VARARGS | METH_KEYWORDS, NULL},
    {"get_f", (PyCFunction) py_rpd_get_f, METH_NOARGS, NULL},
    {"get_xlu", (PyCFunction) py_rpd_get_xlu, METH_VARARGS | METH_KEYWORDS, NULL},
    {"get_clu", (PyCFunction) py_rpd_get_clu, METH_VARARGS | METH_KEYWORDS, NULL},
    {"get_h", (PyCFunction) py_rpd_get_h, METH_VARARGS | METH_KEYWORDS, NULL},
    {"get_a", (PyCFunction) py_rpd_get_a, METH_VARARGS | METH_KEYWORDS, NULL},
    {"get_h_c", (PyCFunction) py_rpd_get_h_c, METH_VARARGS | METH_KEYWORDS, NULL},
    {"get_x_type", (PyCFunction) py_rpd_get_x_type, METH_VARARGS | METH_KEYWORDS, NULL},
    {"get_x", (PyCFunction) py_rpd_get_x, METH_VARARGS | METH_KEYWORDS, NULL},
    {"get_y", (PyCFunction) py_rpd_get_y, METH_VARARGS | METH_KEYWORDS, NULL},
    {"get_z", (PyCFunction) py_rpd_get_z, METH_VARARGS | METH_KEYWORDS, NULL},
    {"information", (PyCFunction) py_rpd_information, METH_NOARGS, NULL},
    {"terminate", (PyCFunction) py_rpd_terminate, METH_NOARGS, NULL},
    {NULL, NULL, 0, NULL}  /* Sentinel */
};

/* rpd python module documentation */

PyDoc_STRVAR(rpd_module_doc,

"The rpd package reads and writes quadratic programming \n"
"(and related) problem data to and from a QPLIB-format data file. \n"
"Variables may be continuous, binary or integer.\n"
"\n"
"See $GALAHAD/html/Python/rpd.html for argument lists, call order\n"
"and other details.\n"
"\n"
);

/* rpd python module definition */
static struct PyModuleDef module = {
   PyModuleDef_HEAD_INIT,
   "rpd",               /* name of module */
   rpd_module_doc,      /* module documentation, may be NULL */
   -1,                  /* size of per-interpreter state of the module,or -1
                           if the module keeps state in global variables */
   rpd_module_methods   /* module methods */
};

/* Python module initialization */
PyMODINIT_FUNC PyInit_rpd(void) { // must be same as module name above
    import_array();  // for NumPy arrays
    return PyModule_Create(&module);
}
