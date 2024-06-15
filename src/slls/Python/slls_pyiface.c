//* \file slls_pyiface.c */

/*
 * THIS VERSION: GALAHAD 4.3 - 2024-01-02 AT 08:30 GMT.
 *
 *-*-*-*-*-*-*-*-*-  GALAHAD_SLLS PYTHON INTERFACE  *-*-*-*-*-*-*-*-*-*-
 *
 *  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
 *  Principal author: Jaroslav Fowkes & Nick Gould
 *
 *  History -
 *   originally released GALAHAD Version 4.3. January 2nd 2024
 *
 *  For full documentation, see
 *   http://galahad.rl.ac.uk/galahad-www/specs.html
 */

#include "galahad_python.h"
#include "galahad_slls.h"

/* Nested SBLS and CONVERT control and inform prototypes */
bool sbls_update_control(struct sbls_control_type *control,
                         PyObject *py_options);
PyObject* sbls_make_options_dict(const struct sbls_control_type *control);
PyObject* sbls_make_inform_dict(const struct sbls_inform_type *inform);
bool convert_update_control(struct convert_control_type *control,
                        PyObject *py_options);
PyObject* convert_make_options_dict(const struct convert_control_type *control);
PyObject* convert_make_inform_dict(const struct convert_inform_type *inform);

/* Module global variables */
static void *data;                       // private internal data
static struct slls_control_type control; // control struct
static struct slls_inform_type inform;   // inform struct
static bool init_called = false;         // record if initialise was called
static int status = 0;                   // exit status

//  *-*-*-*-*-*-*-*-*-*-   UPDATE CONTROL    -*-*-*-*-*-*-*-*-*-*

/* Update the control options: use C defaults but update any passed via Python*/
static bool slls_update_control(struct slls_control_type *control,
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
        if(strcmp(key_name, "stop_d") == 0){
            if(!parse_double_option(value, "stop_d",
                                  &control->stop_d))
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
        if(strcmp(key_name, "convert_options") == 0){
            if(!convert_update_control(&control->convert_control, value))
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
PyObject* slls_make_options_dict(const struct slls_control_type *control){
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
    PyDict_SetItemString(py_options, "stop_d",
                         PyFloat_FromDouble(control->stop_d));
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
    PyDict_SetItemString(py_options, "convert_options",
                         convert_make_options_dict(&control->convert_control));

    return py_options;
}


//  *-*-*-*-*-*-*-*-*-*-   MAKE TIME    -*-*-*-*-*-*-*-*-*-*

/* Take the time struct from C and turn it into a python dictionary */
static PyObject* slls_make_time_dict(const struct slls_time_type *time){
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
//    PyDict_SetItemString(py_time, "clock_total",
//                         PyFloat_FromDouble(time->clock_total));
//    PyDict_SetItemString(py_time, "clock_analyse",
//                         PyFloat_FromDouble(time->clock_analyse));
//    PyDict_SetItemString(py_time, "clock_factorize",
//                         PyFloat_FromDouble(time->clock_factorize));
//    PyDict_SetItemString(py_time, "clock_solve",
//                         PyFloat_FromDouble(time->clock_solve));

    return py_time;
}

//  *-*-*-*-*-*-*-*-*-*-   MAKE INFORM    -*-*-*-*-*-*-*-*-*-*

/* Take the inform struct from C and turn it into a python dictionary */
static PyObject* slls_make_inform_dict(const struct slls_inform_type *inform){
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
                         slls_make_time_dict(&inform->time));

    // Set dictionaries from subservient packages
    PyDict_SetItemString(py_inform, "sbls_inform",
                         sbls_make_inform_dict(&inform->sbls_inform));
    PyDict_SetItemString(py_inform, "convert_inform",
                         convert_make_inform_dict(&inform->convert_inform));

    return py_inform;
}

//  *-*-*-*-*-*-*-*-*-*-   SLLS_INITIALIZE    -*-*-*-*-*-*-*-*-*-*

static PyObject* py_slls_initialize(PyObject *self){

    // Call slls_initialize
    slls_initialize(&data, &control, &status);

    // Record that SLLS has been initialised
    init_called = true;

    // Return options Python dictionary
    PyObject *py_options = slls_make_options_dict(&control);
    return Py_BuildValue("O", py_options);
}

//  *-*-*-*-*-*-*-*-*-*-*-*-   SLLS_LOAD    -*-*-*-*-*-*-*-*-*-*-*-*

static PyObject* py_slls_load(PyObject *self, PyObject *args, PyObject *keywds){
    PyArrayObject *py_Ao_row, *py_Ao_col, *py_Ao_ptr;
    PyObject *py_options = NULL;
    int *Ao_row = NULL, *Ao_col = NULL, *Ao_ptr = NULL;
    const char *Ao_type;
    int n, o, Ao_ne, Ao_ptr_ne;

    // Check that package has been initialised
    if(!check_init(init_called))
        return NULL;

    // Parse positional and keyword arguments
    static char *kwlist[] = {"n","o","Ao_type","Ao_ne","Ao_row",
                             "Ao_col","Ao_ptr_ne","Ao_ptr",
                             "options",NULL};

    if(!PyArg_ParseTupleAndKeywords(args, keywds, "iisiOOiO|O",
                                    kwlist, &n, &o,
                                    &Ao_type, &Ao_ne, &py_Ao_row,
                                    &py_Ao_col, &Ao_ptr_ne, &py_Ao_ptr,
                                    &py_options))
        return NULL;

    // Check that array inputs are of correct type, size, and shape

    if(!(
        check_array_int("Ao_row", py_Ao_row, Ao_ne) &&
        check_array_int("Ao_col", py_Ao_col, Ao_ne) &&
        check_array_int("Ao_ptr", py_Ao_ptr, Ao_ptr_ne)
        ))
        return NULL;

    // Convert 64bit integer Ao_row array to 32bit
    if((PyObject *) py_Ao_row != Py_None){
        Ao_row = malloc(Ao_ne * sizeof(int));
        long int *Ao_row_long = (long int *) PyArray_DATA(py_Ao_row);
        for(int i = 0; i < Ao_ne; i++) Ao_row[i] = (int) Ao_row_long[i];
    }

    // Convert 64bit integer Ao_col array to 32bit
    if((PyObject *) py_Ao_col != Py_None){
        Ao_col = malloc(Ao_ne * sizeof(int));
        long int *Ao_col_long = (long int *) PyArray_DATA(py_Ao_col);
        for(int i = 0; i < Ao_ne; i++) Ao_col[i] = (int) Ao_col_long[i];
    }

    // Convert 64bit integer Ao_ptr array to 32bit
    if((PyObject *) py_Ao_ptr != Py_None){
        Ao_ptr = malloc(Ao_ptr_ne * sizeof(int));
        long int *Ao_ptr_long = (long int *) PyArray_DATA(py_Ao_ptr);
        for(int i = 0; i < Ao_ptr_ne; i++) Ao_ptr[i] = (int) Ao_ptr_long[i];
    }

    // Reset control options
    slls_reset_control(&control, &data, &status);

    // Update SLLS control options
    if(!slls_update_control(&control, py_options))
        return NULL;

    // Call slls_import
    slls_import(&control, &data, &status, n, o,
                Ao_type, Ao_ne, Ao_row, Ao_col, Ao_ptr_ne, Ao_ptr);

    // Free allocated memory
    if(Ao_row != NULL) free(Ao_row);
    if(Ao_col != NULL) free(Ao_col);
    if(Ao_ptr != NULL) free(Ao_ptr);

    // Raise any status errors
    if(!check_error_codes(status))
        return NULL;

    // Return None boilerplate
    Py_INCREF(Py_None);
    return Py_None;
}

//  *-*-*-*-*-*-*-*-*-*-   SLLS_SOLVE_LS   -*-*-*-*-*-*-*-*

static PyObject* py_slls_solve_ls(PyObject *self, PyObject *args, PyObject *keywds){
    PyArrayObject *py_Ao_val;
    PyArrayObject *py_b, *py_x, *py_z;
    double *Ao_val, *b, *x, *z;
    int n, o, Ao_ne;

    // Check that package has been initialised
    if(!check_init(init_called))
        return NULL;

    // Parse positional arguments
    static char *kwlist[] = {"n", "o", "Ao_ne", "Ao_val", "b", "x", "z", NULL};
    if(!PyArg_ParseTupleAndKeywords(args, keywds, "iiiOOOO", kwlist, &n, &o,
                                    &Ao_ne, &py_Ao_val, &py_b, &py_x, &py_z))
        return NULL;

    // Check that array inputs are of correct type, size, and shape
    if(!check_array_double("A_val", py_Ao_val, Ao_ne))
        return NULL;
    if(!check_array_double("b", py_b, o))
        return NULL;
    if(!check_array_double("x", py_x, n))
        return NULL;
    if(!check_array_double("z", py_z, n))
        return NULL;

    // Get array data pointer
    Ao_val = (double *) PyArray_DATA(py_Ao_val);
    b = (double *) PyArray_DATA(py_b);
    x = (double *) PyArray_DATA(py_x);
    z = (double *) PyArray_DATA(py_z);

   // Create NumPy output arrays
    npy_intp ndim[] = {n}; // size of g and x_stat
    npy_intp odim[] = {o}; // size of c
    PyArrayObject *py_r =
      (PyArrayObject *) PyArray_SimpleNew(1, odim, NPY_DOUBLE);
    double *r = (double *) PyArray_DATA(py_r);
    PyArrayObject *py_g =
      (PyArrayObject *) PyArray_SimpleNew(1, ndim, NPY_DOUBLE);
    double *g = (double *) PyArray_DATA(py_g);
    PyArrayObject *py_x_stat =
      (PyArrayObject *) PyArray_SimpleNew(1, ndim, NPY_INT);
    int *x_stat = (int *) PyArray_DATA(py_x_stat);

    // Call slls_solve_direct
    status = 1; // set status to 1 on entry
    slls_solve_given_a(&data, NULL, &status, n, o, Ao_ne, Ao_val,
                       b, x, z, r, g, x_stat, NULL);
    // for( int i = 0; i < n; i++) printf("x %f\n", x[i]);
    // for( int i = 0; i < o; i++) printf("c %f\n", c[i]);
    // for( int i = 0; i < n; i++) printf("x_stat %i\n", x_stat[i]);
    // for( int i = 0; i < o; i++) printf("c_stat %i\n", c_stat[i]);

    // Propagate any errors with the callback function
    if(PyErr_Occurred())
        return NULL;

    // Raise any status errors
    if(!check_error_codes(status))
        return NULL;

    // Return x, z, r, g and x_stat
    PyObject *solve_ls_return;
    solve_ls_return = Py_BuildValue("OOOOO", py_x, py_z, py_r, py_g,
                                     py_x_stat);
    Py_INCREF(solve_ls_return);
    return solve_ls_return;
}

//  *-*-*-*-*-*-*-*-*-*-   SLLS_INFORMATION   -*-*-*-*-*-*-*-*

static PyObject* py_slls_information(PyObject *self){

    // Check that package has been initialised
    if(!check_init(init_called))
        return NULL;

    // Call slls_information
    slls_information(&data, &inform, &status);
    // Return status and inform Python dictionary
    PyObject *py_inform = slls_make_inform_dict(&inform);
    return Py_BuildValue("O", py_inform);
}

//  *-*-*-*-*-*-*-*-*-*-   SLLS_TERMINATE   -*-*-*-*-*-*-*-*-*-*

static PyObject* py_slls_terminate(PyObject *self){

    // Check that package has been initialised
    if(!check_init(init_called))
        return NULL;

    // Call slls_terminate
    slls_terminate(&data, &control, &inform);

    // Return None boilerplate
    Py_INCREF(Py_None);
    return Py_None;
}

//  *-*-*-*-*-*-*-*-*-*-   INITIALIZE SLLS PYTHON MODULE    -*-*-*-*-*-*-*-*-*-*

/* slls python module method table */
static PyMethodDef slls_module_methods[] = {
    {"initialize", (PyCFunction) py_slls_initialize, METH_NOARGS, NULL},
    {"load", (PyCFunction) py_slls_load, METH_VARARGS | METH_KEYWORDS, NULL},
    {"solve_ls", (PyCFunction) py_slls_solve_ls, METH_VARARGS | METH_KEYWORDS, NULL},
    {"information", (PyCFunction) py_slls_information, METH_NOARGS, NULL},
    {"terminate", (PyCFunction) py_slls_terminate, METH_NOARGS, NULL},
    {NULL, NULL, 0, NULL}  /* Sentinel */
};

/* slls python module documentation */

PyDoc_STRVAR(slls_module_doc,

"The slls package uses a preconditioned project-gradient method to\n"
"solve a given bound-constrained linear least-squares problem.\n"
"The aim is to minimize the regularized linear least-squares\n"
"objective function\n"
"q(x) =  1/2 || A_o x - b||_W^2 + sigma/2 ||x||_2^2 \n"
"where x is required to lie in the regular simplex\n"
"e^T x = 1, x >= 0,\n"
"where the o by n matrix A_o, the vector \n"
"b and the non-negative weights w and \n"
"sigma are given, e is the vector of ones\n"
"and where the Euclidean and weighted-Euclidean norms\n"
"are given by ||v||_2^2 = v^T v and ||v||_W^2 = v^T W v,\n"
"respectively, with W = diag(w). Any of the components of \n"
"The method offers the choice of direct and iterative solution of the key\n"
"regularization subproblems, and is most suitable for problems\n"
"involving a large number of unknowns x.\n"
"\n"
"See $GALAHAD/html/Python/slls.html for argument lists, call order\n"
"and other details.\n"
"\n"
);

/* slls python module definition */
static struct PyModuleDef module = {
   PyModuleDef_HEAD_INIT,
   "slls",               /* name of module */
   slls_module_doc,      /* module documentation, may be NULL */
   -1,                  /* size of per-interpreter state of the module,or -1
                           if the module keeps state in global variables */
   slls_module_methods   /* module methods */
};

/* Python module initialization */
PyMODINIT_FUNC PyInit_slls(void) { // must be same as module name above
    import_array();  // for NumPy arrays
    return PyModule_Create(&module);
}
