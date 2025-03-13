# Accelerating K-means Clustering Through CUDA Parallel Computing
This repository provides a high-performance implementation of the K-Means clustering algorithm, optimized for execution on NVIDIA GPUs using CUDA. The project focuses on leveraging shared memory and parallel reduction techniques to achieve significant performance improvements over traditional CPU-based approaches.

## Prerequisites
The project requires the following components:
- NVIDIA GPU with Compute Capability 6.0 or higher
- CUDA Toolkit (minimum version 11.0)
- C++ compiler supporting CUDA (such as `nvcc` from the CUDA Toolkit)

## Setup and Usage
1. Clone the repository:
  ```bash
  git clone lorenzo-27/kmeans-cuda
  cd kmeans-cuda
  ```
2. Configure the algorithm parameters:
   - Open `kmeans_config.py`
   - Adjust the clustering parameters according to your requirements
3. Compile the project:
   - If using CLion with CUDA support, the build process is automatically handled.
   - For manual compilation, ensure you create a `cmake-build-release` directory or update the executable path in `kmeans.py`
4. Run the program:
  - Use the Python script kmeans.py to execute the compiled binary and manage datasets and results.

> [!NOTE]
> Upon execution, the program automatically creates two directories:
> - data/: Contains generated datasets
> - results/: Stores performance plots and analysis tables

## Documentation
For a comprehensive understanding of the implementation and performance analysis, please refer to our detailed technical report available <a href="https://github.com/lorenzo-27/kmeans-cuda/blob/master/TR2-kmeans.pdf">here</a>. The report includes:
- Implementation details
- Performance benchmarks
- Experimental results and analysis

## License
This project is licensed under the <a href="https://github.com/lorenzo-27/kmeans-cuda/blob/master/LICENSE" target="_blank">MIT</a> License.
