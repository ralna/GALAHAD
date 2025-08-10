// include guard
#ifndef GALAHAD_C_COMMON_H
#define GALAHAD_C_COMMON_H

// C interface for BSC
struct bsc_control_type {
    bool f_indexing;
    int32_t error;
    int32_t out;
    int32_t print_level;
    int32_t max_col;
    int32_t new_a;
    int32_t extra_space_s;
    bool s_also_by_column;
    bool space_critical;
    bool deallocate_error_fatal;
    char prefix[31];
};

struct bsc_control_type_64 {
    bool f_indexing;
    int64_t error;
    int64_t out;
    int64_t print_level;
    int64_t max_col;
    int64_t new_a;
    int64_t extra_space_s;
    bool s_also_by_column;
    bool space_critical;
    bool deallocate_error_fatal;
    char prefix[31];
};

// C interface for CONVERT
struct convert_control_type {
    bool f_indexing;
    int32_t error;
    int32_t out;
    int32_t print_level;
    bool transpose;
    bool sum_duplicates;
    bool order;
    bool space_critical;
    bool deallocate_error_fatal;
    char prefix[31];
};

struct convert_control_type_64 {
    bool f_indexing;
    int64_t error;
    int64_t out;
    int64_t print_level;
    bool transpose;
    bool sum_duplicates;
    bool order;
    bool space_critical;
    bool deallocate_error_fatal;
    char prefix[31];
};

// C interface for FIT
struct fit_control_type {
    bool f_indexing;
    int32_t error;
    int32_t out;
    int32_t print_level;
    bool space_critical;
    bool deallocate_error_fatal;
    char prefix[31];
};

struct fit_control_type_64 {
    bool f_indexing;
    int64_t error;
    int64_t out;
    int64_t print_level;
    bool space_critical;
    bool deallocate_error_fatal;
    char prefix[31];
};

struct fit_inform_type {
    int32_t status;
    int32_t alloc_status;
    char bad_alloc[81];
};

struct fit_inform_type_64 {
    int64_t status;
    int64_t alloc_status;
    char bad_alloc[81];
};

// C interface for GLS
struct gls_sinfo_type {
    int32_t flag;
    int32_t more;
    int32_t stat;
};

struct gls_sinfo_type_64 {
    int64_t flag;
    int64_t more;
    int64_t stat;
};

// C interface for HASH
struct hash_control_type {
    int32_t error;
    int32_t out;
    int32_t print_level;
    bool space_critical;
    bool deallocate_error_fatal;
    char prefix[31];
};

struct hash_control_type_64 {
    int64_t error;
    int64_t out;
    int64_t print_level;
    bool space_critical;
    bool deallocate_error_fatal;
    char prefix[31];
};

struct hash_inform_type {
    int32_t status;
    int32_t alloc_status;
    char bad_alloc[81];
};

struct hash_inform_type_64 {
    int64_t status;
    int64_t alloc_status;
    char bad_alloc[81];
};

// C interface for HSL
struct ma48_sinfo {
    int32_t flag;
    int32_t more;
    int32_t stat;
};

struct ma48_sinfo_64 {
    int64_t flag;
    int64_t more;
    int64_t stat;
};

struct mc64_control {
    int32_t f_arrays;
    int32_t lp;
    int32_t wp;
    int32_t sp;
    int32_t ldiag;
    int32_t checking;
};

struct mc64_control_64 {
    int64_t f_arrays;
    int64_t lp;
    int64_t wp;
    int64_t sp;
    int64_t ldiag;
    int64_t checking;
};

struct mc64_info {
    int32_t flag;
    int32_t more;
    int32_t strucrank;
    int32_t stat;
};

struct mc64_info_64 {
    int64_t flag;
    int64_t more;
    int64_t strucrank;
    int64_t stat;
};

struct mc68_control {
    int32_t f_array_in;
    int32_t f_array_out;
    int32_t min_l_workspace;
    int32_t lp;
    int32_t wp;
    int32_t mp;
    int32_t nemin;
    int32_t print_level;
    int32_t row_full_thresh;
    int32_t row_search;
};

struct mc68_control_64 {
    int64_t f_array_in;
    int64_t f_array_out;
    int64_t min_l_workspace;
    int64_t lp;
    int64_t wp;
    int64_t mp;
    int64_t nemin;
    int64_t print_level;
    int64_t row_full_thresh;
    int64_t row_search;
};

struct mc68_info {
    int32_t flag;
    int32_t iostat;
    int32_t stat;
    int32_t out_range;
    int32_t duplicate;
    int32_t n_compressions;
    int32_t n_zero_eigs;
    int64_t l_workspace;
    int32_t zb01_info;
    int32_t n_dense_rows;
};

struct mc68_info_64 {
    int64_t flag;
    int64_t iostat;
    int64_t stat;
    int64_t out_range;
    int64_t duplicate;
    int64_t n_compressions;
    int64_t n_zero_eigs;
    int64_t l_workspace;
    int64_t zb01_info;
    int64_t n_dense_rows;
};

// C interface for LHS
struct lhs_control_type {
    int32_t error;
    int32_t out;
    int32_t print_level;
    int32_t duplication;
    bool space_critical;
    bool deallocate_error_fatal;
    char prefix[31];
};

struct lhs_control_type_64 {
    int64_t error;
    int64_t out;
    int64_t print_level;
    int64_t duplication;
    bool space_critical;
    bool deallocate_error_fatal;
    char prefix[31];
};

struct lhs_inform_type {
    int32_t status;
    int32_t alloc_status;
    char bad_alloc[81];
};

struct lhs_inform_type_64 {
    int64_t status;
    int64_t alloc_status;
    char bad_alloc[81];
};

// C interface for LMS
struct lms_control_type {
    bool f_indexing;
    int32_t error;
    int32_t out;
    int32_t print_level;
    int32_t memory_length;
    int32_t method;
    bool any_method;
    bool space_critical;
    bool deallocate_error_fatal;
    char prefix[31];
};

struct lms_control_type_64 {
    bool f_indexing;
    int64_t error;
    int64_t out;
    int64_t print_level;
    int64_t memory_length;
    int64_t method;
    bool any_method;
    bool space_critical;
    bool deallocate_error_fatal;
    char prefix[31];
};

// C interface for NODEND
struct nodend_control_type {
    bool f_indexing;
    char version[31];
    int32_t error;
    int32_t out;
    int32_t print_level;
    bool no_metis_4_use_5_instead;
    char prefix[31];
    int32_t metis4_ptype;
    int32_t metis4_ctype;
    int32_t metis4_itype;
    int32_t metis4_rtype;
    int32_t metis4_dbglvl;
    int32_t metis4_oflags;
    int32_t metis4_pfactor;
    int32_t metis4_nseps;
    int32_t metis5_ptype;
    int32_t metis5_objtype;
    int32_t metis5_ctype;
    int32_t metis5_iptype;
    int32_t metis5_rtype;
    int32_t metis5_dbglvl;
    int32_t metis5_niter;
    int32_t metis5_ncuts;
    int32_t metis5_seed;
    int32_t metis5_no2hop;
    int32_t metis5_minconn;
    int32_t metis5_contig;
    int32_t metis5_compress;
    int32_t metis5_ccorder;
    int32_t metis5_pfactor;
    int32_t metis5_nseps;
    int32_t metis5_ufactor;
    int32_t metis5_niparts;
    int32_t metis5_ondisk;
    int32_t metis5_dropedges;
    int32_t metis5_twohop;
    int32_t metis5_fast;
};

struct nodend_control_type_64 {
    bool f_indexing;
    char version[31];
    int64_t error;
    int64_t out;
    int64_t print_level;
    bool no_metis_4_use_5_instead;
    char prefix[31];
    int64_t metis4_ptype;
    int64_t metis4_ctype;
    int64_t metis4_itype;
    int64_t metis4_rtype;
    int64_t metis4_dbglvl;
    int64_t metis4_oflags;
    int64_t metis4_pfactor;
    int64_t metis4_nseps;
    int64_t metis5_ptype;
    int64_t metis5_objtype;
    int64_t metis5_ctype;
    int64_t metis5_iptype;
    int64_t metis5_rtype;
    int64_t metis5_dbglvl;
    int64_t metis5_niter;
    int64_t metis5_ncuts;
    int64_t metis5_seed;
    int64_t metis5_no2hop;
    int64_t metis5_minconn;
    int64_t metis5_contig;
    int64_t metis5_compress;
    int64_t metis5_ccorder;
    int64_t metis5_pfactor;
    int64_t metis5_nseps;
    int64_t metis5_ufactor;
    int64_t metis5_niparts;
    int64_t metis5_ondisk;
    int64_t metis5_dropedges;
    int64_t metis5_twohop;
    int64_t metis5_fast;
};

// C interface for PRESOLVE
struct presolve_inform_type {
    int32_t status;
    int32_t status_continue;
    int32_t status_continued;
    int32_t nbr_transforms;
    char message[3][81];
};

struct presolve_inform_type_64 {
    int64_t status;
    int64_t status_continue;
    int64_t status_continued;
    int64_t nbr_transforms;
    char message[3][81];
};

// C interface for ROOTS
struct roots_inform_type {
    int32_t status;
    int32_t alloc_status;
    char bad_alloc[81];
};

struct roots_inform_type_64 {
    int64_t status;
    int64_t alloc_status;
    char bad_alloc[81];
};

// C interface for RPD
struct rpd_control_type {
    bool f_indexing;
    int32_t qplib;
    int32_t error;
    int32_t out;
    int32_t print_level;
    bool space_critical;
    bool deallocate_error_fatal;
};

struct rpd_control_type_64 {
    bool f_indexing;
    int64_t qplib;
    int64_t error;
    int64_t out;
    int64_t print_level;
    bool space_critical;
    bool deallocate_error_fatal;
};

struct rpd_inform_type {
    int32_t status;
    int32_t alloc_status;
    int32_t io_status;
    int32_t line;
    char p_type[4];
    char bad_alloc[81];
};

struct rpd_inform_type_64 {
    int64_t status;
    int64_t alloc_status;
    int64_t io_status;
    int64_t line;
    char p_type[4];
    char bad_alloc[81];
};

// C interface for SCU
struct scu_control_type {
    bool f_indexing;
};

struct scu_inform_type {
    int32_t status;
    int32_t alloc_status;
    int32_t inertia[3];
};

struct scu_inform_type_64 {
    int64_t status;
    int64_t alloc_status;
    int64_t inertia[3];
};

// C interface for SEC
struct sec_inform_type {
    int32_t status;
};

struct sec_inform_type_64 {
    int64_t status;
};

// C interface for SHA
struct sha_control_type {
    bool f_indexing;
    int32_t error;
    int32_t out;
    int32_t print_level;
    int32_t approximation_algorithm;
    int32_t dense_linear_solver;
    int32_t extra_differences;
    int32_t sparse_row;
    int32_t recursion_max;
    int32_t recursion_entries_required;
    bool average_off_diagonals;
    bool space_critical;
    bool deallocate_error_fatal;
    char prefix[31];
};

struct sha_control_type_64 {
    bool f_indexing;
    int64_t error;
    int64_t out;
    int64_t print_level;
    int64_t approximation_algorithm;
    int64_t dense_linear_solver;
    int64_t extra_differences;
    int64_t sparse_row;
    int64_t recursion_max;
    int64_t recursion_entries_required;
    bool average_off_diagonals;
    bool space_critical;
    bool deallocate_error_fatal;
    char prefix[31];
};

// C interface for BQP
struct bqp_time_type {
    float total;
    float analyse;
    float factorize;
    float solve;
};

// C interface for SLLS
struct slls_time_type {
    float total;
    float analyse;
    float factorize;
    float solve;
};

// C interface for VERSION
void version_galahad(int32_t *major, int32_t *minor, int32_t *patch);
void version_galahad_64(int64_t *major, int64_t *minor, int64_t *patch);

// end include guard
#endif
