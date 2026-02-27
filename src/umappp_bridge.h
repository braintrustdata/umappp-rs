#ifndef UMAPPP_BRIDGE_H
#define UMAPPP_BRIDGE_H

#include <cstddef>
#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

struct UmapppOptions {
    double local_connectivity;
    double bandwidth;
    double mix_ratio;
    double spread;
    double min_dist;
    double a;
    double b;
    uint8_t has_a;
    uint8_t has_b;
    double repulsion_strength;
    uint8_t initialize_method;
    uint8_t initialize_random_on_spectral_fail;
    double initialize_spectral_scale;
    uint8_t initialize_spectral_jitter;
    double initialize_spectral_jitter_sd;
    double initialize_random_scale;
    uint64_t initialize_seed;
    int32_t num_epochs;
    uint8_t has_num_epochs;
    double learning_rate;
    double negative_sample_rate;
    int32_t num_neighbors;
    uint64_t optimize_seed;
    int32_t num_threads;
    uint8_t parallel_optimization;
};

void umappp_default_options(UmapppOptions* out);

void* umappp_initialize(
    const double* data,
    size_t data_dim,
    int32_t num_obs,
    size_t num_dim,
    double* embedding,
    const UmapppOptions* options);

int umappp_status_run(void* status, double* embedding, int epoch_limit);
int umappp_status_run_full(void* status, double* embedding);
int umappp_status_epoch(void* status);
int umappp_status_num_epochs(void* status);
size_t umappp_status_num_dimensions(void* status);
int32_t umappp_status_num_observations(void* status);
void umappp_status_free(void* status);

const char* umappp_last_error(void);
void umappp_clear_last_error(void);

int umappp_run_reference(
    const double* data,
    size_t data_dim,
    int32_t num_obs,
    size_t num_dim,
    const UmapppOptions* options,
    double* embedding);

// Fit UMAP, accepting input in row-major layout (num_obs x data_dim) and producing
// output in row-major layout (num_obs x num_dim).
int umappp_fit_rowmajor(
    const double* data_rowmajor,
    size_t data_dim,
    int32_t num_obs,
    size_t num_dim,
    const UmapppOptions* options,
    double* embedding_rowmajor);

// Fit UMAP using a precomputed kNN graph.
//
// - indices/distances are shaped (num_obs, k) in row-major order.
// - embedding_rowmajor is shaped (num_obs, num_dim) in row-major order.
int umappp_fit_from_knn(
    const uint32_t* indices,
    const double* distances,
    size_t k,
    int32_t num_obs,
    size_t num_dim,
    const UmapppOptions* options,
    double* embedding_rowmajor);

// Float32 variant of fit_from_knn to reduce memory traffic in optimization-heavy paths.
int umappp_fit_from_knn_f32(
    const uint32_t* indices,
    const float* distances,
    size_t k,
    int32_t num_obs,
    size_t num_dim,
    const UmapppOptions* options,
    float* embedding_rowmajor);

// Transform new points into an existing embedding using a precomputed kNN graph.
//
// - indices/distances are shaped (num_new, k) in row-major order, with indices
//   referring to rows in the training embedding.
// - train_embedding_rowmajor is shaped (num_train, num_dim) in row-major order.
// - embedding_rowmajor is shaped (num_new, num_dim) in row-major order.
int umappp_transform_from_knn(
    const uint32_t* indices,
    const double* distances,
    size_t k,
    int32_t num_new,
    int32_t num_train,
    size_t num_dim,
    const double* train_embedding_rowmajor,
    const UmapppOptions* options,
    double* embedding_rowmajor);

#ifdef __cplusplus
} // extern "C"
#endif

#endif
