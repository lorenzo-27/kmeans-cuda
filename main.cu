#include <algorithm>
#include <cfloat>
#include <chrono>
#include <vector>
#include <random>
#include <iostream>
#include <fstream>
#include <cuda_runtime.h>

// Utility function for CUDA error checking
#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error in %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
}

struct DatasetBase {
    size_t n_points;
    size_t n_dims;
};

struct DatasetSoA : DatasetBase {
    float* data{}; // Device pointer
    std::vector<float> h_data; // Host data

    void allocate(size_t points, size_t dims) {
        n_points = points;
        n_dims = dims;
        h_data.resize(points * dims);
        CHECK_CUDA(cudaMalloc(&data, points * dims * sizeof(float)));
    }

    void copyToDevice() const {
        CHECK_CUDA(cudaMemcpy(data, h_data.data(), h_data.size() * sizeof(float), cudaMemcpyHostToDevice));
    }

    void free() const {
        if (data) CHECK_CUDA(cudaFree(data));
    }

    float& at(size_t dim, size_t point) {
        return h_data[dim * n_points + point];
    }

    [[nodiscard]] const float& at(size_t dim, size_t point) const {
        return h_data[dim * n_points + point];
    }
};

struct CentroidsBase {
    size_t k{};
    size_t n_dims{};
    int* d_counts{}; // Device pointer for counts
    std::vector<size_t> h_counts; // Host counts for sequential version
    int* h_counts_cuda{}; // Host counts for CUDA version
};

struct CentroidsSoA : CentroidsBase {
    float* data{}; // Device pointer
    std::vector<float> h_data; // Host data for sequential version
    float* h_data_cuda{}; // Host data for CUDA version

    void allocate(size_t clusters, size_t dims) {
        k = clusters;
        n_dims = dims;
        h_data.resize(clusters * dims);
        h_counts.resize(clusters);
        
        h_data_cuda = new float[clusters * dims];
        h_counts_cuda = new int[clusters];
        
        CHECK_CUDA(cudaMalloc(&data, clusters * dims * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_counts, clusters * sizeof(int)));
    }

    void copyToDevice() const {
        CHECK_CUDA(cudaMemcpy(data, h_data_cuda, k * n_dims * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_counts, h_counts_cuda, k * sizeof(int), cudaMemcpyHostToDevice));
    }

    void copyToHost() const {
        CHECK_CUDA(cudaMemcpy(h_data_cuda, data, k * n_dims * sizeof(float), cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(h_counts_cuda, d_counts, k * sizeof(int), cudaMemcpyDeviceToHost));
    }

    void free() const {
        if (data) CHECK_CUDA(cudaFree(data));
        if (d_counts) CHECK_CUDA(cudaFree(d_counts));
        delete[] h_data_cuda;
        delete[] h_counts_cuda;
    }

    float& at(size_t dim, size_t cluster) {
        return h_data[dim * k + cluster];
    }

    [[nodiscard]] const float& at(size_t dim, size_t cluster) const {
        return h_data[dim * k + cluster];
    }
};

// Sequential implementation
float compute_distance(const DatasetSoA& data, size_t point_idx, const CentroidsSoA& centroids, size_t centroid_idx) {
    float dist = 0.0f;
    for (size_t d = 0; d < data.n_dims; ++d) {
        float diff = data.at(d, point_idx) - centroids.at(d, centroid_idx);
        dist += diff * diff;
    }
    return dist;
}

void kmeans_sequential(const DatasetSoA& data, CentroidsSoA& centroids, std::vector<int>& assignments, int max_iter) {
    std::vector<float> new_centroid_data(centroids.k * data.n_dims, 0.0f);

    for (int iter = 0; iter < max_iter; ++iter) {
        // Assignment step
        for (size_t i = 0; i < data.n_points; ++i) {
            float min_dist = std::numeric_limits<float>::max();
            int best_cluster = 0;

            for (size_t j = 0; j < centroids.k; ++j) {
                float dist = compute_distance(data, i, centroids, j);
                if (dist < min_dist) {
                    min_dist = dist;
                    best_cluster = static_cast<int>(j);
                }
            }
            assignments[i] = best_cluster;
        }

        // Reset centroids
        std::fill(new_centroid_data.begin(), new_centroid_data.end(), 0.0f);
        std::fill(centroids.h_counts.begin(), centroids.h_counts.end(), 0);

        // Update step
        for (size_t i = 0; i < data.n_points; ++i) {
            int cluster = assignments[i];
            centroids.h_counts[cluster]++;
            for (size_t d = 0; d < data.n_dims; ++d) {
                new_centroid_data[d * centroids.k + cluster] += data.at(d, i);
            }
        }

        // Normalize centroids
        for (size_t j = 0; j < centroids.k; ++j) {
            if (centroids.h_counts[j] > 0) {
                for (size_t d = 0; d < data.n_dims; ++d) {
                    centroids.at(d, j) = new_centroid_data[d * centroids.k + j] / centroids.h_counts[j];
                }
            }
        }
    }
}

// CUDA kernel for assignment step
__global__ void assignClusters(const float* data, size_t n_points, size_t n_dims,
                             const float* centroids, size_t k,
                             int* assignments, int* counts) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_points) return;

    float min_dist = FLT_MAX;
    int best_cluster = 0;

    // Calculate distances to each centroid
    for (int c = 0; c < k; c++) {
        float dist = 0.0f;
        for (int d = 0; d < n_dims; d++) {
            float diff = data[d * n_points + idx] - centroids[d * k + c];
            dist += diff * diff;
        }
        if (dist < min_dist) {
            min_dist = dist;
            best_cluster = c;
        }
    }

    assignments[idx] = best_cluster;
    atomicAdd(&counts[best_cluster], 1);
}

// CUDA kernel for update step
__global__ void updateCentroids(const float* data, size_t n_points, size_t n_dims,
                                        float* centroids, size_t k,
                                        const int* assignments, const int* counts) {
    extern __shared__ float shared_sums[];

    int cluster = blockIdx.x;
    int dim = blockIdx.y;
    if (cluster >= k || dim >= n_dims) return;

    float sum = 0.0f;
    // Parallelize over points
    for (int i = threadIdx.x; i < n_points; i += blockDim.x) {
        if (assignments[i] == cluster) {
            sum += data[dim * n_points + i];
        }
    }

    // Parallel reduction in shared memory
    shared_sums[threadIdx.x] = sum;
    __syncthreads();

    // Reduce within block
    for (int stride = blockDim.x/2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            shared_sums[threadIdx.x] += shared_sums[threadIdx.x + stride];
        }
        __syncthreads();
    }

    // Write result
    if (threadIdx.x == 0 && counts[cluster] > 0) {
        centroids[dim * k + cluster] = shared_sums[0] / counts[cluster];
    }
}

DatasetSoA read_dataset(const std::string& filename) {
    DatasetSoA data;
    std::ifstream file(filename);
    std::string line;

    // Read header
    std::getline(file, line);
    size_t n_dims = std::count(line.begin(), line.end(), ',') + 1;

    // Count points
    size_t n_points = 1;
    while (std::getline(file, line)) n_points++;

    // Allocate memory
    data.allocate(n_points, n_dims);

    // Reset file pointer and skip header
    file.clear();
    file.seekg(0);
    std::getline(file, line);

    // Read data
    size_t point_idx = 0;
    while (std::getline(file, line)) {
        size_t pos = 0;
        size_t dim_idx = 0;
        while ((pos = line.find(',')) != std::string::npos) {
            data.h_data[dim_idx * n_points + point_idx] = std::stof(line.substr(0, pos));
            line.erase(0, pos + 1);
            dim_idx++;
        }
        data.h_data[dim_idx * n_points + point_idx] = std::stof(line);
        point_idx++;
    }

    data.copyToDevice();
    return data;
}

CentroidsSoA initialize_centroids(const DatasetSoA& data, size_t k) {
    std::mt19937 gen(42);
    std::uniform_int_distribution<> dis(0, data.n_points - 1);

    CentroidsSoA centroids;
    centroids.allocate(k, data.n_dims);

    // Initialize both host and CUDA data with the same values
    for (size_t i = 0; i < k; ++i) {
        size_t idx = dis(gen);
        for (size_t d = 0; d < data.n_dims; ++d) {
            centroids.h_data[d * k + i] = data.h_data[d * data.n_points + idx];
            centroids.h_data_cuda[d * k + i] = centroids.h_data[d * k + i];
        }
    }

    centroids.copyToDevice();
    return centroids;
}

void kmeans_cuda(const DatasetSoA& data, const CentroidsSoA& centroids, int* assignments, int max_iter) {
    // Allocate device memory for assignments
    int* d_assignments;
    CHECK_CUDA(cudaMalloc(&d_assignments, data.n_points * sizeof(int)));

    // Calculate grid and block dimensions
    constexpr int BLOCK_SIZE = 256; // Threads per block
    dim3 assignGrid((data.n_points + BLOCK_SIZE - 1) / BLOCK_SIZE);
    dim3 assignBlock(BLOCK_SIZE);

    dim3 updateGrid(centroids.k, data.n_dims);
    dim3 updateBlock(256);
    size_t sharedMemSize = 256 * sizeof(float);

    for (int iter = 0; iter < max_iter; ++iter) {
        // Reset counts
        CHECK_CUDA(cudaMemset(centroids.d_counts, 0, centroids.k * sizeof(int)));

        // Assignment step
        assignClusters<<<assignGrid, assignBlock>>>(
            data.data, data.n_points, data.n_dims,
            centroids.data, centroids.k,
            d_assignments, centroids.d_counts
        );
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());

        // Update step
        updateCentroids <<<updateGrid, updateBlock, sharedMemSize>>>(
            data.data, data.n_points, data.n_dims,
            centroids.data, centroids.k,
            d_assignments, centroids.d_counts
        );
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());
    }

    // Copy final assignments back to host
    CHECK_CUDA(cudaMemcpy(assignments, d_assignments, data.n_points * sizeof(int), cudaMemcpyDeviceToHost));
    
    // Cleanup
    CHECK_CUDA(cudaFree(d_assignments));
}

int main(int argc, char* argv[]) {
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " <input_file> <k> <max_iter>\n";
        return 1;
    }

    std::string input_file = argv[1];
    int k = std::stoi(argv[2]);
    int max_iter = std::stoi(argv[3]);

    // Read data and initialize
    auto data = read_dataset(input_file);
    auto centroids = initialize_centroids(data, k);
    std::vector<int> assignments(data.n_points);
    std::vector<int> assignments_cuda(data.n_points);

    // Run sequential k-means
    auto seq_start = std::chrono::high_resolution_clock::now();
    kmeans_sequential(data, centroids, assignments, max_iter);
    auto seq_end = std::chrono::high_resolution_clock::now();
    auto seq_time = std::chrono::duration_cast<std::chrono::milliseconds>(seq_end - seq_start).count();
    
    // Run CUDA k-means
    cudaEvent_t cuda_start, cuda_stop;
    cudaEventCreate(&cuda_start);
    cudaEventCreate(&cuda_stop);
    
    cudaEventRecord(cuda_start);
    kmeans_cuda(data, centroids, assignments_cuda.data(), max_iter);
    cudaEventRecord(cuda_stop);
    cudaEventSynchronize(cuda_stop);
    
    float cuda_time;
    cudaEventElapsedTime(&cuda_time, cuda_start, cuda_stop);
    
    // Print results
    std::cout << "Sequential execution time: " << seq_time << "ms\n";
    std::cout << "CUDA execution time: " << cuda_time << "ms\n";
    std::cout << "Speedup: " << static_cast<float>(seq_time) / cuda_time << "x\n";

    // Cleanup
    data.free();
    centroids.free();
    cudaEventDestroy(cuda_start);
    cudaEventDestroy(cuda_stop);

    return 0;
}