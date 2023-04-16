//* \file blls_pyiface.c */

/*
 * THIS VERSION: GALAHAD 4.1 - 2023-04-14 AT 14:10 GMT.
 *
 *-*-*-*-*-*-*-*-*-  GALAHAD_BLLS PYTHON INTERFACE  *-*-*-*-*-*-*-*-*-*-
 *
 *  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
 *  Principal author: Jaroslav Fowkes & Nick Gould
 *
 *  History -
 *   originally released GALAHAD Version 4.1. April 14th 2023
 *
 *  For full documentation, see
 *   http://galahad.rl.ac.uk/galahad-www/specs.html
 */

#include "galahad_python.h"
#include "galahad_blls.h"

/* Nested SBLS and CONVERT control and inform prototypes */
bool sbls_update_control(struct sbls_control_type *control,
                         PyObject *py_options);
PyObject* sbls_make_options_dict(const struct sbls_control_type *control);
PyObject* sbls_make_inform_dict(const struct sbls_inform_type *inform);
//bool convert_update_control(struct convert_control_type *control,
//                        PyObject *py_options);
//PyObject* convert_make_options_dict(const struct convert_control_type *control);
//PyObject* convert_make_inform_dict(const struct convert_inform_type *inform);

/* Module global variables */
static void *data;                       // private internal data
static struct blls_control_type control; // control struct
static struct blls_inform_type inform;   // inform struct
static bool init_called = false;         // record if initialise was called
static int status = 0;                   // exit status

//  *-*-*-*-*-*-*-*-*-*-   UPDATE CONTROL    -*-*-*-*-*-*-*-*-*-*

/* Update the control options: use C defaults but update any passed via Python*/
static bool blls_update_control(struct blls_control_type *control,
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
        if(strcmp(key_name, "preconditioner") == 0){
            if(!parse_int_option(value, "preconditioner",
                                  &control->preconditioner))
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
        if(strcmp(key_name, "arcsearch_max_steps") == 0){
            if(!parse_int_option(value, "arcsearch_max_steps",
                                  &control->arcsearch_max_steps))
                return false;
            continue;
        }
        if(strcmp(key_name, "sif_file_device") == 0){
            if(!parse_int_option(value, "sif_file_device",
                                  &control->sif_file_device))
                return false;
            continue;
        }
        if(strcmp(key_name, "weight") == 0){
            if(!parse_double_option(value, "weight",
                                  &control->weight))
                return false;
            continue;
        }
        if(strcmp(key_name, "infinity") == 0){
            if(!parse_double_option(value, "infinity",
                                  &control->infinity))
                return false;
            continue;
        }
        if(strcmp(key_name, "stop_d") == 0){
            if(!parse_double_option(value, "stop_d",
                                  &control->stop_d))
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
        if(strcmp(key_name, "alpha_max") == 0){
            if(!parse_double_option(value, "alpha_max",
                                  &control->alpha_max))
                return false;
            continue;
        }
        if(strcmp(key_name, "alpha_initial") == 0){
            if(!parse_double_option(value, "alpha_initial",
                                  &control->alpha_initial))
                return false;
            continue;
        }
        if(strcmp(key_name, "alpha_reduction") == 0){
            if(!parse_double_option(value, "alpha_reduction",
                                  &control->alpha_reduction))
                return false;
            continue;
        }
        if(strcmp(key_name, "arcsearch_acceptance_tol") == 0){
            if(!parse_double_option(value, "arcsearch_acceptance_tol",
                                  &control->arcsearch_acceptance_tol))
                return false;
            continue;
        }
        if(strcmp(key_name, "stabilisation_weight") == 0){
            if(!parse_double_option(value, "stabilisation_weight",
                                  &control->stabilisation_weight))
                return false;
            continue;
        }
        if(strcmp(key_name, "cpu_time_limit") == 0){
            if(!parse_double_option(value, "cpu_time_limit",
                                  &control->cpu_time_limit))
                return false;
            continue;
        }
        if(strcmp(key_name, "direct_subproblem_solve") == 0){
            if(!parse_bool_option(value, "direct_subproblem_solve",
                                  &control->direct_subproblem_solve))
                return false;
            continue;
        }
        if(strcmp(key_name, "exact_arc_search") == 0){
            if(!parse_bool_option(value, "exact_arc_search",
                                  &control->exact_arc_search))
                return false;
            continue;
        }
        if(strcmp(key_name, "advance") == 0){
            if(!parse_bool_option(value, "advance",
                                  &control->advance))
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
        //if(strcmp(key_name, "convert_options") == 0){
        //    if(!convert_update_control(&control->convert_control, value))
        //        return false;
        //    continue;
        //}

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
PyObject* blls_make_options_dict(const struct blls_control_type *control){
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
    PyDict_SetItemString(py_options, "preconditioner",
                         PyLong_FromLong(control->preconditioner));
    PyDict_SetItemString(py_options, "ratio_cg_vs_sd",
                         PyLong_FromLong(control->ratio_cg_vs_sd));
    PyDict_SetItemString(py_options, "change_max",
                         PyLong_FromLong(control->change_max));
    PyDict_SetItemString(py_options, "cg_maxit",
                         PyLong_FromLong(control->cg_maxit));
    PyDict_SetItemString(py_options, "arcsearch_max_steps",
                         PyLong_FromLong(control->arcsearch_max_steps));
    PyDict_SetItemString(py_options, "sif_file_device",
                         PyLong_FromLong(control->sif_file_device));
    PyDict_SetItemString(py_options, "weight",
                         PyFloat_FromDouble(control->weight));
    PyDict_SetItemString(py_options, "infinity",
                         PyFloat_FromDouble(control->infinity));
    PyDict_SetItemString(py_options, "stop_d",
                         PyFloat_FromDouble(control->stop_d));
    PyDict_SetItemString(py_options, "identical_bounds_tol",
                         PyFloat_FromDouble(control->identical_bounds_tol));
    PyDict_SetItemString(py_options, "stop_cg_relative",
                         PyFloat_FromDouble(control->stop_cg_relative));
    PyDict_SetItemString(py_options, "stop_cg_absolute",
                         PyFloat_FromDouble(control->stop_cg_absolute));
    PyDict_SetItemString(py_options, "alpha_max",
                         PyFloat_FromDouble(control->alpha_max));
    PyDict_SetItemString(py_options, "alpha_initial",
                         PyFloat_FromDouble(control->alpha_initial));
    PyDict_SetItemString(py_options, "alpha_reduction",
                         PyFloat_FromDouble(control->alpha_reduction));
    PyDict_SetItemString(py_options, "arcsearch_acceptance_tol",
                         PyFloat_FromDouble(control->arcsearch_acceptance_tol));
    PyDict_SetItemString(py_options, "stabilisation_weight",
                         PyFloat_FromDouble(control->stabilisation_weight));
    PyDict_SetItemString(py_options, "cpu_time_limit",
                         PyFloat_FromDouble(control->cpu_time_limit));
    PyDict_SetItemString(py_options, "direct_subproblem_solve",
                         PyBool_FromLong(control->direct_subproblem_solve));
    PyDict_SetItemString(py_options, "exact_arc_search",
                         PyBool_FromLong(control->exact_arc_search));
    PyDict_SetItemString(py_options, "advance",
                         PyBool_FromLong(control->advance));
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
    //PyDict_SetItemString(py_options, "convert_options",
    //                     convert_make_options_dict(&control->convert_control));

    return py_options;
}


//  *-*-*-*-*-*-*-*-*-*-   MAKE TIME    -*-*-*-*-*-*-*-*-*-*

/* Take the time struct from C and turn it into a python dictionary */
static PyObject* blls_make_time_dict(const struct blls_time_type *time){
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
static PyObject* blls_make_inform_dict(const struct blls_inform_type *inform){
    PyObject *py_inform = PyDict_New();

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

    // Set time nested dictionary
    PyDict_SetItemString(py_inform, "time",
                         blls_make_time_dict(&inform->time));

    // Set dictionaries from subservient packages
    PyDict_SetItemString(py_inform, "sbls_inform",
                         sbls_make_inform_dict(&inform->sbls_inform));
    //PyDict_SetItemString(py_inform, "convert_inform",
    //                     convert_make_inform_dict(&inform->convert_inform));

    return py_inform;
}

//  *-*-*-*-*-*-*-*-*-*-   BLLS_INITIALIZE    -*-*-*-*-*-*-*-*-*-*

static PyObject* py_blls_initialize(PyObject *self){

    // Call blls_initialize
    blls_initialize(&data, &control, &status);

    // Record that BLLS has been initialised
    init_called = true;

    // Return options Python dictionary
    PyObject *py_options = blls_make_options_dict(&control);
    return Py_BuildValue("O", py_options);
}

//  *-*-*-*-*-*-*-*-*-*-*-*-   BLLS_LOAD    -*-*-*-*-*-*-*-*-*-*-*-*

static PyObject* py_blls_load(PyObject *self, PyObject *args, PyObject *keywds){
    PyArrayObject *py_A_row, *py_A_col, *py_A_ptr;
    PyObject *py_options = NULL;
    int *A_row = NULL, *A_col = NULL, *A_ptr = NULL;
    const char *A_type;
    int n, m, A_ne;

    // Check that package has been initialised
    if(!check_init(init_called))
        return NULL;

    // Parse positional and keyword arguments
    static char *kwlist[] = {"n","m",
                             "A_type","A_ne","A_row","A_col","A_ptr",
                             "options"};

    if(!PyArg_ParseTupleAndKeywords(args, keywds, "iisiOOOO|O",
                                    kwlist, &n, &m,
                                    &A_type, &A_ne, &py_A_row,
                                    &py_A_col, &py_A_ptr,
                                    &py_options))
        return NULL;

    // Check that array inputs are of correct type, size, and shape

    if(!(
        check_array_int("A_row", py_A_row, A_ne) &&
        check_array_int("A_col", py_A_col, A_ne) &&
        check_array_int("A_ptr", py_A_ptr, n+1)
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
        A_ptr = malloc((n+1) * sizeof(int));
        long int *A_ptr_long = (long int *) PyArray_DATA(py_A_ptr);
        for(int i = 0; i < n+1; i++) A_ptr[i] = (int) A_ptr_long[i];
    }

    // Reset control options
    blls_reset_control(&control, &data, &status);

    // Update BLLS control options
    if(!blls_update_control(&control, py_options))
        return NULL;

    // Call blls_import
    blls_import(&control, &data, &status, n, m,
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

//  *-*-*-*-*-*-*-*-*-*-   BLLS_SOLVE_LS   -*-*-*-*-*-*-*-*

static PyObject* py_blls_solve_ls(PyObject *self, PyObject *args){
    PyArrayObject *py_w, *py_A_val;
    PyArrayObject *py_b, *py_x_l, *py_x_u, *py_x, *py_z;
    double *w, *A_val, *b, *x_l, *x_u, *x, *z;
    int n, m, A_ne;

    // Check that package has been initialised
    if(!check_init(init_called))
        return NULL;

    // Parse positional arguments
    if(!PyArg_ParseTuple(args, "iiOiOOOOOO", &n, &m, &py_w, 
                         &A_ne, &py_A_val, &py_b, &py_x_l, &py_x_u, 
                         &py_x, &py_z))

    // Check that array inputs are of correct type, size, and shape
    if(!check_array_double("w", py_w, m))
        return NULL;
    if(!check_array_double("A_val", py_A_val, A_ne))
        return NULL;
    if(!check_array_double("b", py_b, m))
        return NULL;
    if(!check_array_double("x_l", py_x_l, n))
        return NULL;
    if(!check_array_double("x_u", py_x_u, n))
        return NULL;
    if(!check_array_double("x", py_x, n))
        return NULL;
    if(!check_array_double("z", py_z, n))
        return NULL;
    if(!check_array_double("w", py_w, m))
        return NULL;

    // Get array data pointer
    w = (double *) PyArray_DATA(py_w);
    A_val = (double *) PyArray_DATA(py_A_val);
    b = (double *) PyArray_DATA(py_b);
    x_l = (double *) PyArray_DATA(py_x_l);
    x_u = (double *) PyArray_DATA(py_x_u);
    x = (double *) PyArray_DATA(py_x);
    z = (double *) PyArray_DATA(py_z);
    w = (double *) PyArray_DATA(py_w);

   // Create NumPy output arrays
    npy_intp ndim[] = {n}; // size of x_stat
    npy_intp mdim[] = {m}; // size of c and c_ztar
    PyArrayObject *py_c = 
      (PyArrayObject *) PyArray_SimpleNew(1, mdim, NPY_DOUBLE);
    double *c = (double *) PyArray_DATA(py_c);
    PyArrayObject *py_g = 
      (PyArrayObject *) PyArray_SimpleNew(1, ndim, NPY_DOUBLE);
    double *g = (double *) PyArray_DATA(py_g);
    PyArrayObject *py_x_stat = 
      (PyArrayObject *) PyArray_SimpleNew(1, ndim, NPY_INT);
    int *x_stat = (int *) PyArray_DATA(py_x_stat);

    // Call blls_solve_direct
    status = 1; // set status to 1 on entry
    blls_solve_given_a(&data, NULL, &status, n, m, A_ne, A_val, 
                       b, x_l, x_u, x, z, c, g, x_stat, w, NULL);
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

    // Return x, z, c, g and x_stat
    PyObject *solve_ls_return;
    solve_ls_return = Py_BuildValue("OOOOO", py_x, py_z, py_c, py_g, 
                                     py_x_stat);
    Py_INCREF(solve_ls_return);
    return solve_ls_return;
}

//  *-*-*-*-*-*-*-*-*-*-   BLLS_INFORMATION   -*-*-*-*-*-*-*-*

static PyObject* py_blls_information(PyObject *self){

    // Check that package has been initialised
    if(!check_init(init_called))
        return NULL;

    // Call blls_information
    blls_information(&data, &inform, &status);
    // Return status and inform Python dictionary
    PyObject *py_inform = blls_make_inform_dict(&inform);
    return Py_BuildValue("O", py_inform);
}

//  *-*-*-*-*-*-*-*-*-*-   BLLS_TERMINATE   -*-*-*-*-*-*-*-*-*-*

static PyObject* py_blls_terminate(PyObject *self){

    // Check that package has been initialised
    if(!check_init(init_called))
        return NULL;

    // Call blls_terminate
    blls_terminate(&data, &control, &inform);

    // Return None boilerplate
    Py_INCREF(Py_None);
    return Py_None;
}

//  *-*-*-*-*-*-*-*-*-*-   INITIALIZE BLLS PYTHON MODULE    -*-*-*-*-*-*-*-*-*-*

/* blls python module method table */
static PyMethodDef blls_module_methods[] = {
    {"initialize", (PyCFunction) py_blls_initialize, METH_NOARGS,NULL},
    {"load", (PyCFunction) py_blls_load, METH_VARARGS | METH_KEYWORDS, NULL},
    {"solve_ls", (PyCFunction) py_blls_solve_ls, METH_VARARGS, NULL},
    {"information", (PyCFunction) py_blls_information, METH_NOARGS, NULL},
    {"terminate", (PyCFunction) py_blls_terminate, METH_NOARGS, NULL},
    {NULL, NULL, 0, NULL}  /* Sentinel */
};

/* blls python module documentation */

PyDoc_STRVAR(blls_module_doc,

"The blls package uses a preconditioned project-gradient method to\n"
"solve a given bound-constrained linear least-squares problem.\n"
"The aim is to minimize the regularized linear least-squares\n"
"objective function\n"
"q(x) =  1/2 || A x - b||_W^2 + sigma/2 ||x||_2^2 \n"
"subject to the simple bounds\n"
"x_l <= x <= x_u,\n"
"where the m by n matrix A, the vectors \n"
"b, x_l, x_u and the non-negative weights w and \n"
"sigma are given, and where the Euclidean and weighted-Euclidean norms\n"
"are given by ||v||_2^2 = v^T v and ||v||_W^2 = v^T W v,\n"
"respectively, with W = diag(w). Any of the components of \n"
"the vectors x_l or x_u may be infinite.\n"
"The method offers the choice of direct and iterative solution of the key\n"
"regularization subproblems, and is most suitable for problems\n"
"involving a large number of unknowns x.\n"
"\n"
"See $GALAHAD/html/Python/blls.html for argument lists, call order\n"
"and other details.\n"
"\n"
);

/* blls python module definition */
static struct PyModuleDef module = {
   PyModuleDef_HEAD_INIT,
   "blls",               /* name of module */
   blls_module_doc,      /* module documentation, may be NULL */
   -1,                  /* size of per-interpreter state of the module,or -1
                           if the module keeps state in global variables */
   blls_module_methods   /* module methods */
};

/* Python module initialization */
PyMODINIT_FUNC PyInit_blls(void) { // must be same as module name above
    import_array();  // for NumPy arrays
    return PyModule_Create(&module);
}
