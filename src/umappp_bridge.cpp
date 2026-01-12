#include "umappp_bridge.h"

#include <exception>
#include <limits>
#include <memory>
#include <string>
#include <utility>

#include "knncolle/knncolle.hpp"
#include "umappp/umappp.hpp"

namespace {

thread_local std::string g_last_error;

void clear_error() {
    g_last_error.clear();
}

void set_error(const char* msg) {
    if (msg) {
        g_last_error = msg;
    } else {
        g_last_error = "unknown error";
    }
}


bool fill_options(const UmapppOptions& src, umappp::Options* dst) {
    if (!dst) {
        set_error("options output pointer is null");
        return false;
    }

    if (src.num_neighbors <= 0) {
        set_error("num_neighbors must be positive");
        return false;
    }
    if (src.num_threads <= 0) {
        set_error("num_threads must be positive");
        return false;
    }
    if (src.has_num_epochs && src.num_epochs <= 0) {
        set_error("num_epochs must be positive");
        return false;
    }

    umappp::Options opt;
    opt.local_connectivity = src.local_connectivity;
    opt.bandwidth = src.bandwidth;
    opt.mix_ratio = src.mix_ratio;
    opt.spread = src.spread;
    opt.min_dist = src.min_dist;
    if (src.has_a) {
        opt.a = src.a;
    } else {
        opt.a.reset();
    }
    if (src.has_b) {
        opt.b = src.b;
    } else {
        opt.b.reset();
    }
    opt.repulsion_strength = src.repulsion_strength;

    switch (src.initialize_method) {
        case 0:
            opt.initialize_method = umappp::InitializeMethod::SPECTRAL;
            break;
        case 1:
            opt.initialize_method = umappp::InitializeMethod::RANDOM;
            break;
        case 2:
            opt.initialize_method = umappp::InitializeMethod::NONE;
            break;
        default:
            set_error("initialize_method must be 0 (spectral), 1 (random), or 2 (none)");
            return false;
    }

    opt.initialize_random_on_spectral_fail = (src.initialize_random_on_spectral_fail != 0);
    opt.initialize_spectral_scale = src.initialize_spectral_scale;
    opt.initialize_spectral_jitter = (src.initialize_spectral_jitter != 0);
    opt.initialize_spectral_jitter_sd = src.initialize_spectral_jitter_sd;
    opt.initialize_random_scale = src.initialize_random_scale;
    opt.initialize_seed = src.initialize_seed;
    if (src.has_num_epochs) {
        opt.num_epochs = src.num_epochs;
    } else {
        opt.num_epochs.reset();
    }
    opt.learning_rate = src.learning_rate;
    opt.negative_sample_rate = src.negative_sample_rate;
    opt.num_neighbors = src.num_neighbors;
    opt.optimize_seed = src.optimize_seed;
    opt.num_threads = src.num_threads;
    opt.parallel_optimization = (src.parallel_optimization != 0);

    *dst = std::move(opt);
    return true;
}

} // namespace

extern "C" {

void umappp_default_options(UmapppOptions* out) {
    clear_error();
    if (!out) {
        set_error("options output pointer is null");
        return;
    }

    umappp::Options opt;
    out->local_connectivity = opt.local_connectivity;
    out->bandwidth = opt.bandwidth;
    out->mix_ratio = opt.mix_ratio;
    out->spread = opt.spread;
    out->min_dist = opt.min_dist;
    out->a = opt.a.has_value() ? *(opt.a) : 0.0;
    out->b = opt.b.has_value() ? *(opt.b) : 0.0;
    out->has_a = opt.a.has_value() ? 1 : 0;
    out->has_b = opt.b.has_value() ? 1 : 0;
    out->repulsion_strength = opt.repulsion_strength;
    out->initialize_method = static_cast<uint8_t>(opt.initialize_method);
    out->initialize_random_on_spectral_fail = opt.initialize_random_on_spectral_fail ? 1 : 0;
    out->initialize_spectral_scale = opt.initialize_spectral_scale;
    out->initialize_spectral_jitter = opt.initialize_spectral_jitter ? 1 : 0;
    out->initialize_spectral_jitter_sd = opt.initialize_spectral_jitter_sd;
    out->initialize_random_scale = opt.initialize_random_scale;
    out->initialize_seed = opt.initialize_seed;
    out->num_epochs = opt.num_epochs.has_value() ? *(opt.num_epochs) : 0;
    out->has_num_epochs = opt.num_epochs.has_value() ? 1 : 0;
    out->learning_rate = opt.learning_rate;
    out->negative_sample_rate = opt.negative_sample_rate;
    out->num_neighbors = opt.num_neighbors;
    out->optimize_seed = opt.optimize_seed;
    out->num_threads = opt.num_threads;
    out->parallel_optimization = opt.parallel_optimization ? 1 : 0;
}

void* umappp_initialize(
    const double* data,
    size_t data_dim,
    int32_t num_obs,
    size_t num_dim,
    double* embedding,
    const UmapppOptions* options)
{
    clear_error();
    if (!data || !embedding) {
        set_error("data or embedding pointer is null");
        return nullptr;
    }
    if (!options) {
        set_error("options pointer is null");
        return nullptr;
    }
    if (data_dim == 0 || num_dim == 0) {
        set_error("data_dim and num_dim must be positive");
        return nullptr;
    }
    if (num_obs <= 0) {
        set_error("num_obs must be positive");
        return nullptr;
    }

    umappp::Options opt;
    if (!fill_options(*options, &opt)) {
        return nullptr;
    }

    try {
        auto distance = std::make_shared<knncolle::EuclideanDistance<double, double> >();
        auto builder = knncolle::VptreeBuilder<int, double, double>(distance);
        auto status = new umappp::Status<int, double>(
            umappp::initialize(
                data_dim,
                static_cast<int>(num_obs),
                data,
                builder,
                num_dim,
                embedding,
                opt
            )
        );
        return static_cast<void*>(status);
    } catch (const std::exception& e) {
        set_error(e.what());
    } catch (...) {
        set_error("unknown exception during umappp_initialize");
    }

    return nullptr;
}

int umappp_status_run(void* status, double* embedding, int epoch_limit) {
    clear_error();
    if (!status || !embedding) {
        set_error("status or embedding pointer is null");
        return -1;
    }

    auto* ptr = static_cast<umappp::Status<int, double>*>(status);
    try {
        ptr->run(embedding, epoch_limit);
    } catch (const std::exception& e) {
        set_error(e.what());
        return -1;
    } catch (...) {
        set_error("unknown exception during umappp_status_run");
        return -1;
    }

    return 0;
}

int umappp_status_run_full(void* status, double* embedding) {
    clear_error();
    if (!status || !embedding) {
        set_error("status or embedding pointer is null");
        return -1;
    }

    auto* ptr = static_cast<umappp::Status<int, double>*>(status);
    try {
        ptr->run(embedding);
    } catch (const std::exception& e) {
        set_error(e.what());
        return -1;
    } catch (...) {
        set_error("unknown exception during umappp_status_run_full");
        return -1;
    }

    return 0;
}

int umappp_status_epoch(void* status) {
    clear_error();
    if (!status) {
        set_error("status pointer is null");
        return -1;
    }
    return static_cast<umappp::Status<int, double>*>(status)->epoch();
}

int umappp_status_num_epochs(void* status) {
    clear_error();
    if (!status) {
        set_error("status pointer is null");
        return -1;
    }
    return static_cast<umappp::Status<int, double>*>(status)->num_epochs();
}

size_t umappp_status_num_dimensions(void* status) {
    clear_error();
    if (!status) {
        set_error("status pointer is null");
        return 0;
    }
    return static_cast<umappp::Status<int, double>*>(status)->num_dimensions();
}

int32_t umappp_status_num_observations(void* status) {
    clear_error();
    if (!status) {
        set_error("status pointer is null");
        return -1;
    }
    return static_cast<umappp::Status<int, double>*>(status)->num_observations();
}

void umappp_status_free(void* status) {
    clear_error();
    if (!status) {
        return;
    }
    delete static_cast<umappp::Status<int, double>*>(status);
}

const char* umappp_last_error(void) {
    if (g_last_error.empty()) {
        return nullptr;
    }
    return g_last_error.c_str();
}

void umappp_clear_last_error(void) {
    clear_error();
}

int umappp_run_reference(
    const double* data,
    size_t data_dim,
    int32_t num_obs,
    size_t num_dim,
    const UmapppOptions* options,
    double* embedding)
{
    clear_error();
    if (!data || !embedding) {
        set_error("data or embedding pointer is null");
        return -1;
    }
    if (!options) {
        set_error("options pointer is null");
        return -1;
    }
    if (data_dim == 0 || num_dim == 0) {
        set_error("data_dim and num_dim must be positive");
        return -1;
    }
    if (num_obs <= 0) {
        set_error("num_obs must be positive");
        return -1;
    }

    umappp::Options opt;
    if (!fill_options(*options, &opt)) {
        return -1;
    }

    try {
        auto distance = std::make_shared<knncolle::EuclideanDistance<double, double> >();
        auto builder = knncolle::VptreeBuilder<int, double, double>(distance);
        auto status = umappp::initialize(
            data_dim,
            static_cast<int>(num_obs),
            data,
            builder,
            num_dim,
            embedding,
            opt
        );
        status.run(embedding);
    } catch (const std::exception& e) {
        set_error(e.what());
        return -1;
    } catch (...) {
        set_error("unknown exception during umappp_run_reference");
        return -1;
    }

    return 0;
}

} // extern "C"
