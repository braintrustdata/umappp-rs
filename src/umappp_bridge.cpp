#include "umappp_bridge.h"

#include <exception>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "knncolle/knncolle.hpp"
#include "umappp/neighbor_similarities.hpp"
#include "umappp/optimize_layout.hpp"
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

template<typename Index_, typename Float_, class Rng_>
void optimize_layout_transform(
    std::size_t num_dim,
    Float_* new_embedding,
    const Float_* train_embedding,
    Index_ num_train,
    umappp::EpochData<Index_, Float_>& setup,
    Float_ a,
    Float_ b,
    Float_ gamma,
    Float_ initial_alpha,
    Rng_& rng,
    int epoch_limit)
{
    auto& n = setup.current_epoch;
    const auto num_epochs = setup.total_epochs;
    const Float_ one = static_cast<Float_>(1);
    const Float_ two = static_cast<Float_>(2);
    const Float_ repulsive_distance_offset = static_cast<Float_>(0.001);
    const Float_ attractive_constant = -two * a * b;
    const Float_ repulsive_constant = two * gamma * b;
    const Float_ inv_num_epochs = one / static_cast<Float_>(num_epochs);

    for (; n < epoch_limit; ++n) {
        const Float_ epoch = n;
        const Float_ alpha = initial_alpha * (one - epoch * inv_num_epochs);

        const Index_ num_new = setup.cumulative_num_edges.size() - 1;
        for (Index_ i = 0; i < num_new; ++i) {
            const auto start = setup.cumulative_num_edges[i], end = setup.cumulative_num_edges[i + 1];
            auto left = new_embedding + sanisizer::product_unsafe<std::size_t>(i, num_dim);

            for (auto j = start; j < end; ++j) {
                if (setup.epoch_of_next_sample[j] > epoch) {
                    continue;
                }

                {
                    const auto right = train_embedding + sanisizer::product_unsafe<std::size_t>(setup.edge_targets[j], num_dim);
                    const Float_ dist2 = umappp::quick_squared_distance(left, right, num_dim);
                    const Float_ pd2b = std::pow(dist2, b);
                    const Float_ grad_coef = (attractive_constant * pd2b) / (dist2 * (a * pd2b + one));

                    for (std::size_t d = 0; d < num_dim; ++d) {
                        left[d] += alpha * umappp::clamp(grad_coef * (left[d] - right[d]));
                    }
                }

                const Float_ epochs_per_negative_sample = setup.epochs_per_negative_sample[j];
                const int num_neg_samples = (epoch - setup.epoch_of_next_negative_sample[j]) / epochs_per_negative_sample;

                for (int p = 0; p < num_neg_samples; ++p) {
                    const auto sampled = aarand::discrete_uniform(rng, num_train);
                    const auto right = train_embedding + sanisizer::product_unsafe<std::size_t>(sampled, num_dim);
                    const Float_ dist2 = umappp::quick_squared_distance(left, right, num_dim);
                    const Float_ grad_coef = repulsive_constant / ((repulsive_distance_offset + dist2) * (a * std::pow(dist2, b) + one));

                    for (std::size_t d = 0; d < num_dim; ++d) {
                        left[d] += alpha * umappp::clamp(grad_coef * (left[d] - right[d]));
                    }
                }

                setup.epoch_of_next_sample[j] += setup.epochs_per_sample[j];
                setup.epoch_of_next_negative_sample[j] += num_neg_samples * epochs_per_negative_sample;
            }
        }
    }
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

int umappp_fit_rowmajor(
    const double* data_rowmajor,
    size_t data_dim,
    int32_t num_obs,
    size_t num_dim,
    const UmapppOptions* options,
    double* embedding_rowmajor)
{
    clear_error();
    if (!data_rowmajor || !embedding_rowmajor) {
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
        const size_t nobs = static_cast<size_t>(num_obs);

        // Convert row-major (nobs x data_dim) to column-major (data_dim x nobs)
        std::vector<double> data_colmajor(data_dim * nobs);
        for (size_t i = 0; i < nobs; ++i) {
            for (size_t d = 0; d < data_dim; ++d) {
                data_colmajor[d + data_dim * i] = data_rowmajor[i * data_dim + d];
            }
        }

        // umappp expects column-major embedding: (num_dim rows, nobs cols)
        std::vector<double> embedding_colmajor(num_dim * nobs);

        auto distance = std::make_shared<knncolle::EuclideanDistance<double, double> >();
        auto builder = knncolle::VptreeBuilder<int, double, double>(distance);
        auto status = umappp::initialize(
            data_dim,
            static_cast<int>(num_obs),
            data_colmajor.data(),
            builder,
            num_dim,
            embedding_colmajor.data(),
            opt
        );
        status.run(embedding_colmajor.data());

        // Convert embedding to row-major (nobs x num_dim)
        for (size_t i = 0; i < nobs; ++i) {
            for (size_t d = 0; d < num_dim; ++d) {
                embedding_rowmajor[i * num_dim + d] = embedding_colmajor[d + num_dim * i];
            }
        }

    } catch (const std::exception& e) {
        set_error(e.what());
        return -1;
    } catch (...) {
        set_error("unknown exception during umappp_fit_rowmajor");
        return -1;
    }

    return 0;
}

int umappp_fit_from_knn(
    const uint32_t* indices,
    const double* distances,
    size_t k,
    int32_t num_obs,
    size_t num_dim,
    const UmapppOptions* options,
    double* embedding_rowmajor)
{
    clear_error();
    if (!indices || !distances || !embedding_rowmajor) {
        set_error("indices/distances/embedding pointer is null");
        return -1;
    }
    if (!options) {
        set_error("options pointer is null");
        return -1;
    }
    if (k == 0 || num_dim == 0) {
        set_error("k and num_dim must be positive");
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
        const size_t nobs = static_cast<size_t>(num_obs);

        umappp::NeighborList<uint32_t, double> nl;
        nl.resize(nobs);
        for (size_t i = 0; i < nobs; ++i) {
            auto& row = nl[i];
            row.reserve(k);
            const size_t base = i * k;
            for (size_t j = 0; j < k; ++j) {
                const uint32_t idx = indices[base + j];
                const double dist = distances[base + j];
                if (idx == static_cast<uint32_t>(i)) {
                    continue;
                }
                row.emplace_back(idx, dist);
            }
        }

        // umappp expects column-major embedding: (num_dim rows, nobs cols)
        std::vector<double> embedding_colmajor(num_dim * nobs);
        auto status = umappp::initialize(std::move(nl), num_dim, embedding_colmajor.data(), opt);
        status.run(embedding_colmajor.data());

        // Convert embedding to row-major (nobs x num_dim)
        for (size_t i = 0; i < nobs; ++i) {
            for (size_t d = 0; d < num_dim; ++d) {
                embedding_rowmajor[i * num_dim + d] = embedding_colmajor[d + num_dim * i];
            }
        }

    } catch (const std::exception& e) {
        set_error(e.what());
        return -1;
    } catch (...) {
        set_error("unknown exception during umappp_fit_from_knn");
        return -1;
    }

    return 0;
}

int umappp_fit_from_knn_f32(
    const uint32_t* indices,
    const float* distances,
    size_t k,
    int32_t num_obs,
    size_t num_dim,
    const UmapppOptions* options,
    float* embedding_rowmajor)
{
    clear_error();
    if (!indices || !distances || !embedding_rowmajor) {
        set_error("indices/distances/embedding pointer is null");
        return -1;
    }
    if (!options) {
        set_error("options pointer is null");
        return -1;
    }
    if (k == 0 || num_dim == 0) {
        set_error("k and num_dim must be positive");
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
        const size_t nobs = static_cast<size_t>(num_obs);

        umappp::NeighborList<uint32_t, float> nl;
        nl.resize(nobs);
        for (size_t i = 0; i < nobs; ++i) {
            auto& row = nl[i];
            row.reserve(k);
            const size_t base = i * k;
            for (size_t j = 0; j < k; ++j) {
                const uint32_t idx = indices[base + j];
                const float dist = distances[base + j];
                if (idx == static_cast<uint32_t>(i)) {
                    continue;
                }
                row.emplace_back(idx, dist);
            }
        }

        // umappp expects column-major embedding: (num_dim rows, nobs cols)
        std::vector<float> embedding_colmajor(num_dim * nobs);
        auto status = umappp::initialize(std::move(nl), num_dim, embedding_colmajor.data(), opt);
        status.run(embedding_colmajor.data());

        // Convert embedding to row-major (nobs x num_dim)
        for (size_t i = 0; i < nobs; ++i) {
            for (size_t d = 0; d < num_dim; ++d) {
                embedding_rowmajor[i * num_dim + d] = embedding_colmajor[d + num_dim * i];
            }
        }

    } catch (const std::exception& e) {
        set_error(e.what());
        return -1;
    } catch (...) {
        set_error("unknown exception during umappp_fit_from_knn_f32");
        return -1;
    }

    return 0;
}

int umappp_transform_from_knn(
    const uint32_t* indices,
    const double* distances,
    size_t k,
    int32_t num_new,
    int32_t num_train,
    size_t num_dim,
    const double* train_embedding_rowmajor,
    const UmapppOptions* options,
    double* embedding_rowmajor)
{
    clear_error();
    if (!indices || !distances || !train_embedding_rowmajor || !embedding_rowmajor) {
        set_error("input pointer is null");
        return -1;
    }
    if (!options) {
        set_error("options pointer is null");
        return -1;
    }
    if (num_new <= 0 || num_train <= 0) {
        set_error("num_new and num_train must be positive");
        return -1;
    }
    if (num_dim == 0 || k == 0) {
        set_error("num_dim and k must be positive");
        return -1;
    }

    umappp::Options opt;
    if (!fill_options(*options, &opt)) {
        return -1;
    }

    try {
        umappp::NeighborList<int, double> neighbors(num_new);
        for (int i = 0; i < num_new; ++i) {
            auto& current = neighbors[i];
            current.reserve(k);
            const std::size_t offset = static_cast<std::size_t>(i) * k;
            for (std::size_t j = 0; j < k; ++j) {
                const auto idx = indices[offset + j];
                if (idx >= static_cast<uint32_t>(num_train)) {
                    set_error("knn index out of range for training embedding");
                    return -1;
                }
                current.emplace_back(static_cast<int>(idx), distances[offset + j]);
            }
        }

        umappp::NeighborSimilaritiesOptions<double> nsopt;
        nsopt.local_connectivity = opt.local_connectivity;
        nsopt.bandwidth = opt.bandwidth;
        nsopt.num_threads = opt.num_threads;
        umappp::neighbor_similarities(neighbors, nsopt);

        // Initialize new embeddings as weighted averages of neighbor embeddings.
        for (int i = 0; i < num_new; ++i) {
            const auto& current = neighbors[i];
            double total = 0.0;
            for (const auto& entry : current) {
                total += entry.second;
            }

            auto out = embedding_rowmajor + static_cast<std::size_t>(i) * num_dim;
            if (total <= 0 && !current.empty()) {
                const auto src = train_embedding_rowmajor + static_cast<std::size_t>(current.front().first) * num_dim;
                std::copy_n(src, num_dim, out);
            } else {
                std::fill_n(out, num_dim, 0.0);
                if (total > 0) {
                    for (const auto& entry : current) {
                        const auto src = train_embedding_rowmajor + static_cast<std::size_t>(entry.first) * num_dim;
                        const double weight = entry.second / total;
                        for (std::size_t d = 0; d < num_dim; ++d) {
                            out[d] += weight * src[d];
                        }
                    }
                }
            }
        }

        if (!opt.a.has_value() || !opt.b.has_value()) {
            const auto found = umappp::find_ab(opt.spread, opt.min_dist);
            opt.a = found.first;
            opt.b = found.second;
        }

        const int epochs = umappp::choose_num_epochs<int>(opt.num_epochs, num_new);
        auto epoch_data = umappp::similarities_to_epochs(neighbors, epochs, opt.negative_sample_rate);
        umappp::RngEngine rng(opt.optimize_seed);

        optimize_layout_transform<int, double>(
            num_dim,
            embedding_rowmajor,
            train_embedding_rowmajor,
            num_train,
            epoch_data,
            *(opt.a),
            *(opt.b),
            opt.repulsion_strength,
            opt.learning_rate,
            rng,
            epoch_data.total_epochs
        );
        return 0;
    } catch (const std::exception& e) {
        set_error(e.what());
    } catch (...) {
        set_error("unknown exception during umappp_transform_from_knn");
    }

    return -1;
}

} // extern "C"
