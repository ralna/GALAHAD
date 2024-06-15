//* \file lpa_pyiface.c */

/*
 * THIS VERSION: GALAHAD 4.1 - 2023-05-20 AT 10:30 GMT.
 *
 *-*-*-*-*-*-*-*-*-  GALAHAD_LPA PYTHON INTERFACE  *-*-*-*-*-*-*-*-*-*-
 *
 *  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
 *  Principal author: Jaroslav Fowkes & Nick Gould
 *
 *  History -
 *   originally released GALAHAD Version 4.1. April 4th 2023
 *
 *  For full documentation, see
 *   http://galahad.rl.ac.uk/galahad-www/specs.html
 */

#include "galahad_python.h"
#include "galahad_lpa.h"

/* Nested RPD control and inform prototypes */
PyObject* rpd_make_inform_dict(const struct rpd_inform_type *inform);

/* Module global variables */
static void *data;                       // private internal data
static struct lpa_control_type control;  // control struct
static struct lpa_inform_type inform;    // inform struct
static bool init_called = false;         // record if initialise was called
static int status = 0;                   // exit status

//  *-*-*-*-*-*-*-*-*-*-   UPDATE CONTROL    -*-*-*-*-*-*-*-*-*-*

/* Update the control options: use C defaults but update any passed via Python*/
static bool lpa_update_control(struct lpa_control_type *control,
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
        if(strcmp(key_name, "maxit") == 0){
            if(!parse_int_option(value, "maxit",
                                  &control->maxit))
                return false;
            continue;
        }
        if(strcmp(key_name, "max_iterative_refinements") == 0){
            if(!parse_int_option(value, "max_iterative_refinements",
                                  &control->max_iterative_refinements))
                return false;
            continue;
        }
        if(strcmp(key_name, "min_real_factor_size") == 0){
            if(!parse_int_option(value, "min_real_factor_size",
                                  &control->min_real_factor_size))
                return false;
            continue;
        }
        if(strcmp(key_name, "min_integer_factor_size") == 0){
            if(!parse_int_option(value, "min_integer_factor_size",
                                  &control->min_integer_factor_size))
                return false;
            continue;
        }
        if(strcmp(key_name, "random_number_seed") == 0){
            if(!parse_int_option(value, "random_number_seed",
                                  &control->random_number_seed))
                return false;
            continue;
        }
        if(strcmp(key_name, "sif_file_device") == 0){
            if(!parse_int_option(value, "sif_file_device",
                                  &control->sif_file_device))
                return false;
            continue;
        }
        if(strcmp(key_name, "qplib_file_device") == 0){
            if(!parse_int_option(value, "qplib_file_device",
                                  &control->qplib_file_device))
                return false;
            continue;
        }
        if(strcmp(key_name, "infinity") == 0){
            if(!parse_double_option(value, "infinity",
                                  &control->infinity))
                return false;
            continue;
        }
        if(strcmp(key_name, "tol_data") == 0){
            if(!parse_double_option(value, "tol_data",
                                  &control->tol_data))
                return false;
            continue;
        }
        if(strcmp(key_name, "feas_tol") == 0){
            if(!parse_double_option(value, "feas_tol",
                                  &control->feas_tol))
                return false;
            continue;
        }
        if(strcmp(key_name, "relative_pivot_tolerance") == 0){
            if(!parse_double_option(value, "relative_pivot_tolerance",
                                  &control->relative_pivot_tolerance))
                return false;
            continue;
        }
        if(strcmp(key_name, "growth_limit") == 0){
            if(!parse_double_option(value, "growth_limit",
                                  &control->growth_limit))
                return false;
            continue;
        }
        if(strcmp(key_name, "zero_tolerance") == 0){
            if(!parse_double_option(value, "zero_tolerance",
                                  &control->zero_tolerance))
                return false;
            continue;
        }
        if(strcmp(key_name, "change_tolerance") == 0){
            if(!parse_double_option(value, "change_tolerance",
                                  &control->change_tolerance))
                return false;
            continue;
        }
        if(strcmp(key_name, "identical_bounds_tol") == 0){
            if(!parse_double_option(value, "identical_bounds_tol",
                                  &control->identical_bounds_tol))
                return false;
            continue;
        }
        if(strcmp(key_name, "cpu_time_limit") == 0){
            if(!parse_double_option(value, "cpu_time_limit",
                                  &control->cpu_time_limit))
                return false;
            continue;
        }
        if(strcmp(key_name, "clock_time_limit") == 0){
            if(!parse_double_option(value, "clock_time_limit",
                                  &control->clock_time_limit))
                return false;
            continue;
        }
        if(strcmp(key_name, "scale") == 0){
            if(!parse_bool_option(value, "scale",
                                  &control->scale))
                return false;
            continue;
        }
        if(strcmp(key_name, "dual") == 0){
            if(!parse_bool_option(value, "dual",
                                  &control->dual))
                return false;
            continue;
        }
        if(strcmp(key_name, "warm_start") == 0){
            if(!parse_bool_option(value, "warm_start",
                                  &control->warm_start))
                return false;
            continue;
        }
        if(strcmp(key_name, "steepest_edge") == 0){
            if(!parse_bool_option(value, "steepest_edge",
                                  &control->steepest_edge))
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
        if(strcmp(key_name, "generate_qplib_file") == 0){
            if(!parse_bool_option(value, "generate_qplib_file",
                                  &control->generate_qplib_file))
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
        if(strcmp(key_name, "qplib_file_name") == 0){
            if(!parse_char_option(value, "qplib_file_name",
                                  control->qplib_file_name,
                                  sizeof(control->qplib_file_name)))
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
// NB not static as it is used for nested inform within QP Python interface
PyObject* lpa_make_options_dict(const struct lpa_control_type *control){
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
    PyDict_SetItemString(py_options, "maxit",
                         PyLong_FromLong(control->maxit));
    PyDict_SetItemString(py_options, "max_iterative_refinements",
                         PyLong_FromLong(control->max_iterative_refinements));
    PyDict_SetItemString(py_options, "min_real_factor_size",
                         PyLong_FromLong(control->min_real_factor_size));
    PyDict_SetItemString(py_options, "min_integer_factor_size",
                         PyLong_FromLong(control->min_integer_factor_size));
    PyDict_SetItemString(py_options, "random_number_seed",
                         PyLong_FromLong(control->random_number_seed));
    PyDict_SetItemString(py_options, "sif_file_device",
                         PyLong_FromLong(control->sif_file_device));
    PyDict_SetItemString(py_options, "qplib_file_device",
                         PyLong_FromLong(control->qplib_file_device));
    PyDict_SetItemString(py_options, "infinity",
                         PyFloat_FromDouble(control->infinity));
    PyDict_SetItemString(py_options, "tol_data",
                         PyFloat_FromDouble(control->tol_data));
    PyDict_SetItemString(py_options, "feas_tol",
                         PyFloat_FromDouble(control->feas_tol));
    PyDict_SetItemString(py_options, "relative_pivot_tolerance",
                         PyFloat_FromDouble(control->relative_pivot_tolerance));
    PyDict_SetItemString(py_options, "growth_limit",
                         PyFloat_FromDouble(control->growth_limit));
    PyDict_SetItemString(py_options, "zero_tolerance",
                         PyFloat_FromDouble(control->zero_tolerance));
    PyDict_SetItemString(py_options, "change_tolerance",
                         PyFloat_FromDouble(control->change_tolerance));
    PyDict_SetItemString(py_options, "identical_bounds_tol",
                         PyFloat_FromDouble(control->identical_bounds_tol));
    PyDict_SetItemString(py_options, "cpu_time_limit",
                         PyFloat_FromDouble(control->cpu_time_limit));
    PyDict_SetItemString(py_options, "clock_time_limit",
                         PyFloat_FromDouble(control->clock_time_limit));
    PyDict_SetItemString(py_options, "scale",
                         PyBool_FromLong(control->scale));
    PyDict_SetItemString(py_options, "dual",
                         PyBool_FromLong(control->dual));
    PyDict_SetItemString(py_options, "warm_start",
                         PyBool_FromLong(control->warm_start));
    PyDict_SetItemString(py_options, "steepest_edge",
                         PyBool_FromLong(control->steepest_edge));
    PyDict_SetItemString(py_options, "space_critical",
                         PyBool_FromLong(control->space_critical));
    PyDict_SetItemString(py_options, "deallocate_error_fatal",
                         PyBool_FromLong(control->deallocate_error_fatal));
    PyDict_SetItemString(py_options, "generate_sif_file",
                         PyBool_FromLong(control->generate_sif_file));
    PyDict_SetItemString(py_options, "generate_qplib_file",
                         PyBool_FromLong(control->generate_qplib_file));
    PyDict_SetItemString(py_options, "sif_file_name",
                         PyUnicode_FromString(control->sif_file_name));
    PyDict_SetItemString(py_options, "qplib_file_name",
                         PyUnicode_FromString(control->qplib_file_name));
    PyDict_SetItemString(py_options, "prefix",
                         PyUnicode_FromString(control->prefix));

    return py_options;
}


//  *-*-*-*-*-*-*-*-*-*-   MAKE TIME    -*-*-*-*-*-*-*-*-*-*

/* Take the time struct from C and turn it into a python dictionary */
static PyObject* lpa_make_time_dict(const struct lpa_time_type *time){
    PyObject *py_time = PyDict_New();

    // Set float/double time entries

    PyDict_SetItemString(py_time, "total",
                         PyFloat_FromDouble(time->total));
    PyDict_SetItemString(py_time, "preprocess",
                         PyFloat_FromDouble(time->preprocess));
    PyDict_SetItemString(py_time, "clock_total",
                         PyFloat_FromDouble(time->clock_total));
    PyDict_SetItemString(py_time, "clock_preprocess",
                         PyFloat_FromDouble(time->clock_preprocess));

    return py_time;
}

//  *-*-*-*-*-*-*-*-*-*-   MAKE INFORM    -*-*-*-*-*-*-*-*-*-*

/* Take the inform struct from C and turn it into a python dictionary */
static PyObject* lpa_make_inform_dict(const struct lpa_inform_type *inform){
    PyObject *py_inform = PyDict_New();

    PyDict_SetItemString(py_inform, "status",
                         PyLong_FromLong(inform->status));
    PyDict_SetItemString(py_inform, "alloc_status",
                         PyLong_FromLong(inform->alloc_status));
    PyDict_SetItemString(py_inform, "bad_alloc",
                         PyUnicode_FromString(inform->bad_alloc));
    PyDict_SetItemString(py_inform, "iter",
                         PyLong_FromLong(inform->iter));
    PyDict_SetItemString(py_inform, "la04_job",
                         PyLong_FromLong(inform->la04_job));
    PyDict_SetItemString(py_inform, "la04_job_info",
                         PyLong_FromLong(inform->la04_job_info));
    PyDict_SetItemString(py_inform, "obj",
                         PyFloat_FromDouble(inform->obj));
    PyDict_SetItemString(py_inform, "primal_infeasibility",
                         PyFloat_FromDouble(inform->primal_infeasibility));
    PyDict_SetItemString(py_inform, "feasible",
                         PyBool_FromLong(inform->feasible));
    PyDict_SetItemString(py_inform, "rpd_inform",
                         rpd_make_inform_dict(&inform->rpd_inform));

    // include RINFO array
    npy_intp rdim[] = {40};
    PyArrayObject *py_rinfo =
      (PyArrayObject*) PyArray_SimpleNew(1, rdim, NPY_DOUBLE);
    double *rinfo = (double *) PyArray_DATA(py_rinfo);
    for(int i=0; i<40; i++) rinfo[i] = inform->RINFO[i];
    PyDict_SetItemString(py_inform, "RINFO", (PyObject *) py_rinfo);

    // Set time nested dictionary
    PyDict_SetItemString(py_inform, "time",
                         lpa_make_time_dict(&inform->time));

    return py_inform;
}

//  *-*-*-*-*-*-*-*-*-*-   LPA_INITIALIZE    -*-*-*-*-*-*-*-*-*-*

static PyObject* py_lpa_initialize(PyObject *self){

    // Call lpa_initialize
    lpa_initialize(&data, &control, &status);

    // Record that LPA has been initialised
    init_called = true;

    // Return options Python dictionary
    PyObject *py_options = lpa_make_options_dict(&control);
    return Py_BuildValue("O", py_options);
}

//  *-*-*-*-*-*-*-*-*-*-*-*-   LPA_LOAD    -*-*-*-*-*-*-*-*-*-*-*-*

static PyObject* py_lpa_load(PyObject *self, PyObject *args, PyObject *keywds){
    PyArrayObject *py_A_row, *py_A_col, *py_A_ptr;
    PyObject *py_options = NULL;
    int *A_row = NULL, *A_col = NULL, *A_ptr = NULL;
    const char *A_type;
    int n, m, A_ne;

    // Check that package has been initialised
    if(!check_init(init_called))
        return NULL;
    // Parse positional and keyword arguments
    static char *kwlist[] = {"n","m","A_type","A_ne","A_row","A_col","A_ptr",
                             "options",NULL};
    if(!PyArg_ParseTupleAndKeywords(args, keywds, "iisiOOO|O",
                                    kwlist, &n, &m,
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
    lpa_reset_control(&control, &data, &status);

    // Update LPA control options
    if(!lpa_update_control(&control, py_options))
        return NULL;
printf("here\n");
    // Call lpa_import
    lpa_import(&control, &data, &status, n, m,
               A_type, A_ne, A_row, A_col, A_ptr);

    // Free allocated memory
    if(A_row != NULL) free(A_row);
    if(A_col != NULL) free(A_col);
    if(A_ptr != NULL) free(A_ptr);

    // Raise any status errors
    if(!check_error_codes(status))
        return NULL;

    // Return None boilerplate
    Py_INCREF(Py_None);
    return Py_None;
}

//  *-*-*-*-*-*-*-*-*-*-   LPA_SOLVE_LP   -*-*-*-*-*-*-*-*

static PyObject* py_lpa_solve_lp(PyObject *self, PyObject *args, PyObject *keywds){
    PyArrayObject *py_g, *py_A_val;
    PyArrayObject *py_c_l, *py_c_u, *py_x_l, *py_x_u;
    PyArrayObject *py_x, *py_y, *py_z;
    double *g, *A_val, *c_l, *c_u, *x_l, *x_u, *x, *y, *z;
    int n, m, A_ne;
    double f;

    // Check that package has been initialised
    if(!check_init(init_called))
        return NULL;

    // Parse positional arguments
    static char *kwlist[] = {"n", "m", "f", "g", "A_ne", "A_val",
                             "c_l", "c_u", "x_l", "x_u", "x", "y", "z", NULL};
    if(!PyArg_ParseTupleAndKeywords(args, keywds, "iidOiOOOOOOOO", kwlist, &n, &m, &f, &py_g,
                                    &A_ne, &py_A_val, &py_c_l, &py_c_u, &py_x_l, &py_x_u,
                                    &py_x, &py_y, &py_z))
        return NULL;

    // Check that array inputs are of correct type, size, and shape
    if(!check_array_double("g", py_g, n))
        return NULL;
    if(!check_array_double("A_val", py_A_val, A_ne))
        return NULL;
    if(!check_array_double("c_l", py_c_l, m))
        return NULL;
    if(!check_array_double("c_u", py_c_u, m))
        return NULL;
    if(!check_array_double("x_l", py_x_l, n))
        return NULL;
    if(!check_array_double("x_u", py_x_u, n))
        return NULL;
    if(!check_array_double("x", py_x, n))
        return NULL;
    if(!check_array_double("y", py_y, m))
        return NULL;
    if(!check_array_double("z", py_z, n))
        return NULL;

    // Get array data pointer
    g = (double *) PyArray_DATA(py_g);
    A_val = (double *) PyArray_DATA(py_A_val);
    c_l = (double *) PyArray_DATA(py_c_l);
    c_u = (double *) PyArray_DATA(py_c_u);
    x_l = (double *) PyArray_DATA(py_x_l);
    x_u = (double *) PyArray_DATA(py_x_u);
    x = (double *) PyArray_DATA(py_x);
    y = (double *) PyArray_DATA(py_y);
    z = (double *) PyArray_DATA(py_z);

   // Create NumPy output arrays
    npy_intp ndim[] = {n}; // size of x_stat
    npy_intp mdim[] = {m}; // size of c and c_ztar
    PyArrayObject *py_c =
      (PyArrayObject *) PyArray_SimpleNew(1, mdim, NPY_DOUBLE);
    double *c = (double *) PyArray_DATA(py_c);
    PyArrayObject *py_x_stat =
      (PyArrayObject *) PyArray_SimpleNew(1, ndim, NPY_INT);
    int *x_stat = (int *) PyArray_DATA(py_x_stat);
    PyArrayObject *py_c_stat =
      (PyArrayObject *) PyArray_SimpleNew(1, mdim, NPY_INT);
    int *c_stat = (int *) PyArray_DATA(py_c_stat);

    // Call lpa_solve_direct
    status = 1; // set status to 1 on entry
    lpa_solve_lp(&data, &status, n, m, g, f, A_ne, A_val,
                 c_l, c_u, x_l, x_u, x, c, y, z, x_stat, c_stat);
    // for( int i = 0; i < n; i++) printf("x %f\n", x[i]);
    // for( int i = 0; i < m; i++) printf("c %f\n", c[i]);
    // for( int i = 0; i < n; i++) printf("x_stat %i\n", x_stat[i]);
    // for( int i = 0; i < m; i++) printf("c_stat %i\n", c_stat[i]);

    // Propagate any errors with the callback function
    if(PyErr_Occurred())
        return NULL;

    // Raise any status errors
    if(!check_error_codes(status))
        return NULL;

    // Return x, c, y, z, x_stat and c_stat
    PyObject *solve_lp_return;

    // solve_lp_return = Py_BuildValue("O", py_x);
    solve_lp_return = Py_BuildValue("OOOOOO", py_x, py_c, py_y, py_z,
                                              py_x_stat, py_c_stat);
    Py_INCREF(solve_lp_return);
    return solve_lp_return;
}

//  *-*-*-*-*-*-*-*-*-*-   LPA_INFORMATION   -*-*-*-*-*-*-*-*

static PyObject* py_lpa_information(PyObject *self){

    // Check that package has been initialised
    if(!check_init(init_called))
        return NULL;

    // Call lpa_information
    lpa_information(&data, &inform, &status);

    // Return status and inform Python dictionary
    PyObject *py_inform = lpa_make_inform_dict(&inform);
    return Py_BuildValue("O", py_inform);
}

//  *-*-*-*-*-*-*-*-*-*-   LPA_TERMINATE   -*-*-*-*-*-*-*-*-*-*

static PyObject* py_lpa_terminate(PyObject *self){

    // Check that package has been initialised
    if(!check_init(init_called))
        return NULL;

    // Call lpa_terminate
    lpa_terminate(&data, &control, &inform);

    // Return None boilerplate
    Py_INCREF(Py_None);
    return Py_None;
}

//  *-*-*-*-*-*-*-*-*-*-   INITIALIZE LPA PYTHON MODULE    -*-*-*-*-*-*-*-*-*-*

/* lpa python module method table */
static PyMethodDef lpa_module_methods[] = {
    {"initialize", (PyCFunction) py_lpa_initialize, METH_NOARGS, NULL},
    {"load", (PyCFunction) py_lpa_load, METH_VARARGS | METH_KEYWORDS, NULL},
    {"solve_lp", (PyCFunction) py_lpa_solve_lp, METH_VARARGS | METH_KEYWORDS, NULL},
    {"information", (PyCFunction) py_lpa_information, METH_NOARGS, NULL},
    {"terminate", (PyCFunction) py_lpa_terminate, METH_NOARGS, NULL},
    {NULL, NULL, 0, NULL}  /* Sentinel */
};

/* lpa python module documentation */

PyDoc_STRVAR(lpa_module_doc,

"The lpa package uses the simplex method to solve the \n"
"linear programming problem to minimize the linear objective\n"
"l(x) =  g^T x + f\n"
"subject to the general linear constraints\n"
"c_i^l  <=  a_i^Tx  <= c_i^u, i = 1, ... , m\n"
"and the simple bound constraints\n"
"x_j^l  <=  x_j  <= x_j^u, j = 1, ... , n,\n"
"where the vectors g, a_i, c^l, c^u, x^l, x^u and the scalar f are given.\n"
"Any of the constraint bounds c_i^l, c_i^u,\n"
"x_j^l and x_j^u may be infinite.\n"
"Full advantage is taken of any zero coefficients in the \n"
"matrix A of vectors a_i.\n"
"\n"
"See $GALAHAD/html/Python/lpa.html for argument lists, call order\n"
"and other details.\n"
"\n"
"N.B. The package is simply a sophisticated interface to the \n"
"HSL package LA04, and requires that a user has obtained the latter. \n"
"LA04 is not included in GALAHAD but is available without charge to \n"
"recognised academics, see http://www.hsl.rl.ac.uk/catalogue/la04.html.\n"
"If LA04 is unavailable, the interior-point linear programming package \n"
"lpb is an alternative.\n"
"\n"
);

/* lpa python module definition */
static struct PyModuleDef module = {
   PyModuleDef_HEAD_INIT,
   "lpa",               /* name of module */
   lpa_module_doc,      /* module documentation, may be NULL */
   -1,                  /* size of per-interpreter state of the module,or -1
                           if the module keeps state in global variables */
   lpa_module_methods   /* module methods */
};

/* Python module initialization */
PyMODINIT_FUNC PyInit_lpa(void) { // must be same as module name above
    import_array();  // for NumPy arrays
    return PyModule_Create(&module);
}
