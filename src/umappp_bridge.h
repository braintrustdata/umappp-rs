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

#ifdef __cplusplus
} // extern "C"
#endif

#endif
