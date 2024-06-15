//* \file bqp_pyiface.c */

/*
 * THIS VERSION: GALAHAD 4.1 - 2023-05-20 AT 10:30 GMT.
 *
 *-*-*-*-*-*-*-*-*-  GALAHAD_BQP PYTHON INTERFACE  *-*-*-*-*-*-*-*-*-*-
 *
 *  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
 *  Principal author: Jaroslav Fowkes & Nick Gould
 *
 *  History -
 *   originally released GALAHAD Version 4.1. April 5th 2023
 *
 *  For full documentation, see
 *   http://galahad.rl.ac.uk/galahad-www/specs.html
 */

#include "galahad_python.h"
#include "galahad_bqp.h"

/* Nested SBLS control and inform prototypes */
bool sbls_update_control(struct sbls_control_type *control,
                         PyObject *py_options);
PyObject* sbls_make_options_dict(const struct sbls_control_type *control);
PyObject* sbls_make_inform_dict(const struct sbls_inform_type *inform);

/* Module global variables */
static void *data;                       // private internal data
static struct bqp_control_type control;  // control struct
static struct bqp_inform_type inform;    // inform struct
static bool init_called = false;         // record if initialise was called
static int status = 0;                   // exit status

//  *-*-*-*-*-*-*-*-*-*-   UPDATE CONTROL    -*-*-*-*-*-*-*-*-*-*

/* Update the control options: use C defaults but update any passed via Python*/
static bool bqp_update_control(struct bqp_control_type *control,
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
        if(strcmp(key_name, "cold_start") == 0){
            if(!parse_int_option(value, "cold_start",
                                  &control->cold_start))
                return false;
            continue;
        }
        if(strcmp(key_name, "ratio_cg_vs_sd") == 0){
            if(!parse_int_option(value, "ratio_cg_vs_sd",
                                  &control->ratio_cg_vs_sd))
                return false;
            continue;
        }
        if(strcmp(key_name, "change_max") == 0){
            if(!parse_int_option(value, "change_max",
                                  &control->change_max))
                return false;
            continue;
        }
        if(strcmp(key_name, "cg_maxit") == 0){
            if(!parse_int_option(value, "cg_maxit",
                                  &control->cg_maxit))
                return false;
            continue;
        }
        if(strcmp(key_name, "sif_file_device") == 0){
            if(!parse_int_option(value, "sif_file_device",
                                  &control->sif_file_device))
                return false;
            continue;
        }
        if(strcmp(key_name, "infinity") == 0){
            if(!parse_double_option(value, "infinity",
                                  &control->infinity))
                return false;
            continue;
        }
        if(strcmp(key_name, "stop_p") == 0){
            if(!parse_double_option(value, "stop_p",
                                  &control->stop_p))
                return false;
            continue;
        }
        if(strcmp(key_name, "stop_d") == 0){
            if(!parse_double_option(value, "stop_d",
                                  &control->stop_d))
                return false;
            continue;
        }
        if(strcmp(key_name, "stop_c") == 0){
            if(!parse_double_option(value, "stop_c",
                                  &control->stop_c))
                return false;
            continue;
        }
        if(strcmp(key_name, "identical_bounds_tol") == 0){
            if(!parse_double_option(value, "identical_bounds_tol",
                                  &control->identical_bounds_tol))
                return false;
            continue;
        }
        if(strcmp(key_name, "stop_cg_relative") == 0){
            if(!parse_double_option(value, "stop_cg_relative",
                                  &control->stop_cg_relative))
                return false;
            continue;
        }
        if(strcmp(key_name, "stop_cg_absolute") == 0){
            if(!parse_double_option(value, "stop_cg_absolute",
                                  &control->stop_cg_absolute))
                return false;
            continue;
        }
        if(strcmp(key_name, "zero_curvature") == 0){
            if(!parse_double_option(value, "zero_curvature",
                                  &control->zero_curvature))
                return false;
            continue;
        }
        if(strcmp(key_name, "cpu_time_limit") == 0){
            if(!parse_double_option(value, "cpu_time_limit",
                                  &control->cpu_time_limit))
                return false;
            continue;
        }
        if(strcmp(key_name, "exact_arcsearch") == 0){
            if(!parse_bool_option(value, "exact_arcsearch",
                                  &control->exact_arcsearch))
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
        if(strcmp(key_name, "generate_sif_file") == 0){
            if(!parse_bool_option(value, "generate_sif_file",
                                  &control->generate_sif_file))
                return false;
            continue;
        }
        if(strcmp(key_name, "sif_file_name") == 0){
            if(!parse_char_option(value, "sif_file_name",
                                  control->sif_file_name,
                                  sizeof(control->sif_file_name)))
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

        // Otherwise unrecognised option
        PyErr_Format(PyExc_ValueError,
          "unrecognised option options['%s']\n", key_name);
        return false;
    }

    return true; // success
}

//  *-*-*-*-*-*-*-*-*-*-   MAKE OPTIONS    -*-*-*-*-*-*-*-*-*-*

/* Take the control struct from C and turn it into a python options dict */
// NB not static as it is used for nested inform within QP Python interface
PyObject* bqp_make_options_dict(const struct bqp_control_type *control){
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
    PyDict_SetItemString(py_options, "cold_start",
                         PyLong_FromLong(control->cold_start));
    PyDict_SetItemString(py_options, "ratio_cg_vs_sd",
                         PyLong_FromLong(control->ratio_cg_vs_sd));
    PyDict_SetItemString(py_options, "change_max",
                         PyLong_FromLong(control->change_max));
    PyDict_SetItemString(py_options, "cg_maxit",
                         PyLong_FromLong(control->cg_maxit));
    PyDict_SetItemString(py_options, "sif_file_device",
                         PyLong_FromLong(control->sif_file_device));
    PyDict_SetItemString(py_options, "infinity",
                         PyFloat_FromDouble(control->infinity));
    PyDict_SetItemString(py_options, "stop_p",
                         PyFloat_FromDouble(control->stop_p));
    PyDict_SetItemString(py_options, "stop_d",
                         PyFloat_FromDouble(control->stop_d));
    PyDict_SetItemString(py_options, "stop_c",
                         PyFloat_FromDouble(control->stop_c));
    PyDict_SetItemString(py_options, "identical_bounds_tol",
                         PyFloat_FromDouble(control->identical_bounds_tol));
    PyDict_SetItemString(py_options, "stop_cg_relative",
                         PyFloat_FromDouble(control->stop_cg_relative));
    PyDict_SetItemString(py_options, "stop_cg_absolute",
                         PyFloat_FromDouble(control->stop_cg_absolute));
    PyDict_SetItemString(py_options, "zero_curvature",
                         PyFloat_FromDouble(control->zero_curvature));
    PyDict_SetItemString(py_options, "cpu_time_limit",
                         PyFloat_FromDouble(control->cpu_time_limit));
    PyDict_SetItemString(py_options, "exact_arcsearch",
                         PyBool_FromLong(control->exact_arcsearch));
    PyDict_SetItemString(py_options, "space_critical",
                         PyBool_FromLong(control->space_critical));
    PyDict_SetItemString(py_options, "deallocate_error_fatal",
                         PyBool_FromLong(control->deallocate_error_fatal));
    PyDict_SetItemString(py_options, "generate_sif_file",
                         PyBool_FromLong(control->generate_sif_file));
    PyDict_SetItemString(py_options, "sif_file_name",
                         PyUnicode_FromString(control->sif_file_name));
    PyDict_SetItemString(py_options, "prefix",
                         PyUnicode_FromString(control->prefix));
    PyDict_SetItemString(py_options, "sbls_options",
                         sbls_make_options_dict(&control->sbls_control));

    return py_options;
}


//  *-*-*-*-*-*-*-*-*-*-   MAKE TIME    -*-*-*-*-*-*-*-*-*-*

/* Take the time struct from C and turn it into a python dictionary */
static PyObject* bqp_make_time_dict(const struct bqp_time_type *time){
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

    return py_time;
}

//  *-*-*-*-*-*-*-*-*-*-   MAKE INFORM    -*-*-*-*-*-*-*-*-*-*

/* Take the inform struct from C and turn it into a python dictionary */
static PyObject* bqp_make_inform_dict(const struct bqp_inform_type *inform){
    PyObject *py_inform = PyDict_New();

    // Set dictionaries from subservient packages
    PyDict_SetItemString(py_inform, "sbls_inform",
                        sbls_make_inform_dict(&inform->sbls_inform));
    PyDict_SetItemString(py_inform, "status",
                         PyLong_FromLong(inform->status));
    PyDict_SetItemString(py_inform, "alloc_status",
                         PyLong_FromLong(inform->alloc_status));
    PyDict_SetItemString(py_inform, "factorization_status",
                         PyLong_FromLong(inform->factorization_status));
    PyDict_SetItemString(py_inform, "iter",
                         PyLong_FromLong(inform->iter));
    PyDict_SetItemString(py_inform, "cg_iter",
                         PyLong_FromLong(inform->cg_iter));
    PyDict_SetItemString(py_inform, "obj",
                         PyFloat_FromDouble(inform->obj));
    PyDict_SetItemString(py_inform, "norm_pg",
                         PyFloat_FromDouble(inform->norm_pg));
    PyDict_SetItemString(py_inform, "bad_alloc",
                         PyUnicode_FromString(inform->bad_alloc));
    PyDict_SetItemString(py_inform, "sbls_inform",
                         sbls_make_inform_dict(&inform->sbls_inform));

    // Set time nested dictionary
    PyDict_SetItemString(py_inform, "time",
                         bqp_make_time_dict(&inform->time));

    return py_inform;
}

//  *-*-*-*-*-*-*-*-*-*-   BQP_INITIALIZE    -*-*-*-*-*-*-*-*-*-*

static PyObject* py_bqp_initialize(PyObject *self){

    // Call bqp_initialize
    bqp_initialize(&data, &control, &status);

    // Record that BQP has been initialised
    init_called = true;

    // Return options Python dictionary
    PyObject *py_options = bqp_make_options_dict(&control);
    return Py_BuildValue("O", py_options);
}

//  *-*-*-*-*-*-*-*-*-*-*-*-   BQP_LOAD    -*-*-*-*-*-*-*-*-*-*-*-*

static PyObject* py_bqp_load(PyObject *self, PyObject *args, PyObject *keywds){
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

    if(!PyArg_ParseTupleAndKeywords(args, keywds, "isiOOO|O",
                                    kwlist, &n, &H_type, &H_ne, &py_H_row,
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
    bqp_reset_control(&control, &data, &status);

    // Update BQP control options
    if(!bqp_update_control(&control, py_options))
        return NULL;

    // Call bqp_import
    bqp_import(&control, &data, &status, n,
               H_type, H_ne, H_row, H_col, H_ptr);

    // Free allocated memory
    if(H_row != NULL) free(H_row);
    if(H_col != NULL) free(H_col);
    if(H_ptr != NULL) free(H_ptr);

    // Raise any status errors
    if(!check_error_codes(status))
        return NULL;

    // Return None boilerplate
    Py_INCREF(Py_None);
    return Py_None;
}

//  *-*-*-*-*-*-*-*-*-*-   BQP_SOLVE_QP   -*-*-*-*-*-*-*-*

static PyObject* py_bqp_solve_qp(PyObject *self, PyObject *args, PyObject *keywds){
    PyArrayObject *py_g, *py_H_val;
    PyArrayObject *py_x_l, *py_x_u;
    PyArrayObject *py_x, *py_z;
    double *g, *H_val, *x_l, *x_u, *x, *z;
    int n, H_ne;
    double f;

    // Check that package has been initialised
    if(!check_init(init_called))
        return NULL;

    // Parse positional arguments
    static char *kwlist[] = {"n","f","g","H_ne","H_val","x_l","x_u","x","z",NULL};
    if(!PyArg_ParseTupleAndKeywords(args, keywds, "idOiOOOOO", kwlist, &n, &f, &py_g,
                                    &H_ne, &py_H_val, &py_x_l, &py_x_u,
                                    &py_x, &py_z))
        return NULL;

    // Check that array inputs are of correct type, size, and shape
    if(!check_array_double("g", py_g, n))
        return NULL;
    if(!check_array_double("H_val", py_H_val, H_ne))
        return NULL;
    if(!check_array_double("x_l", py_x_l, n))
        return NULL;
    if(!check_array_double("x_u", py_x_u, n))
        return NULL;
    if(!check_array_double("x", py_x, n))
        return NULL;
    if(!check_array_double("z", py_z, n))
        return NULL;

    // Get array data pointer
    g = (double *) PyArray_DATA(py_g);
    H_val = (double *) PyArray_DATA(py_H_val);
    x_l = (double *) PyArray_DATA(py_x_l);
    x_u = (double *) PyArray_DATA(py_x_u);
    x = (double *) PyArray_DATA(py_x);
    z = (double *) PyArray_DATA(py_z);

   // Create NumPy output arrays
    npy_intp ndim[] = {n}; // size of x_stat
    PyArrayObject *py_x_stat =
      (PyArrayObject *) PyArray_SimpleNew(1, ndim, NPY_INT);
    int *x_stat = (int *) PyArray_DATA(py_x_stat);

    // Call bqp_solve_direct
    status = 1; // set status to 1 on entry
    bqp_solve_given_h(&data, &status, n, H_ne, H_val, g, f,
                      x_l, x_u, x, z, x_stat);
    // for( int i = 0; i < n; i++) printf("x %f\n", x[i]);
    // for( int i = 0; i < n; i++) printf("x_stat %i\n", x_stat[i]);

    // Propagate any errors with the callback function
    if(PyErr_Occurred())
        return NULL;

    // Raise any status errors
    if(!check_error_codes(status))
        return NULL;

    // Return x, z and x_stat
    PyObject *solve_qp_return;
    solve_qp_return = Py_BuildValue("OOO", py_x, py_z, py_x_stat);
    Py_INCREF(solve_qp_return);
    return solve_qp_return;

}

//  *-*-*-*-*-*-*-*-*-*-   BQP_INFORMATION   -*-*-*-*-*-*-*-*

static PyObject* py_bqp_information(PyObject *self){

    // Check that package has been initialised
    if(!check_init(init_called))
        return NULL;

    // Call bqp_information
    bqp_information(&data, &inform, &status);

    // Return status and inform Python dictionary
    PyObject *py_inform = bqp_make_inform_dict(&inform);
    return Py_BuildValue("O", py_inform);
}

//  *-*-*-*-*-*-*-*-*-*-   BQP_TERMINATE   -*-*-*-*-*-*-*-*-*-*

static PyObject* py_bqp_terminate(PyObject *self){

    // Check that package has been initialised
    if(!check_init(init_called))
        return NULL;

    // Call bqp_terminate
    bqp_terminate(&data, &control, &inform);

    // Return None boilerplate
    Py_INCREF(Py_None);
    return Py_None;
}

//  *-*-*-*-*-*-*-*-*-*-   INITIALIZE BQP PYTHON MODULE    -*-*-*-*-*-*-*-*-*-*

/* bqp python module method table */
static PyMethodDef bqp_module_methods[] = {
    {"initialize", (PyCFunction) py_bqp_initialize, METH_NOARGS, NULL},
    {"load", (PyCFunction) py_bqp_load, METH_VARARGS | METH_KEYWORDS, NULL},
    {"solve_qp", (PyCFunction) py_bqp_solve_qp, METH_VARARGS | METH_KEYWORDS, NULL},
    {"information", (PyCFunction) py_bqp_information, METH_NOARGS, NULL},
    {"terminate", (PyCFunction) py_bqp_terminate, METH_NOARGS, NULL},
    {NULL, NULL, 0, NULL}  /* Sentinel */
};

/* bqp python module documentation */

PyDoc_STRVAR(bqp_module_doc,

"The bqp package uses a preconditioned, projected-gradient method to solve \n"
"a given bon-constrained convex quadratic programming problem\n"
"The aim is to minimize the quadratic objective\n"
"subject to the simple bound constraints\n"
"x_j^l  <=  x_j  <= x_j^u, j = 1, ... , n,\n"
"where the n by n symmetric, positive-semi-definite matrix\n"
"H, the vectors g, x^l, x^u and the scalar f are given.\n"
"Any of the constraint bounds x_j^l and x_j^u may be infinite.\n"
"The method offers the choice of direct and iterative solution of the key\n"
"regularization subproblems, and is most suitable for problems\n"
"involving a large number of unknowns x.\n"
"\n"
"See $GALAHAD/html/Python/bqp.html for argument lists, call order\n"
"and other details.\n"
"\n"
);

/* bqp python module definition */
static struct PyModuleDef module = {
   PyModuleDef_HEAD_INIT,
   "bqp",               /* name of module */
   bqp_module_doc,      /* module documentation, may be NULL */
   -1,                  /* size of per-interpreter state of the module,or -1
                           if the module keeps state in global variables */
   bqp_module_methods   /* module methods */
};

/* Python module initialization */
PyMODINIT_FUNC PyInit_bqp(void) { // must be same as module name above
    import_array();  // for NumPy arrays
    return PyModule_Create(&module);
}
