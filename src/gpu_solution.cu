#include "algorithm_choices.h"

#include <chrono>    // for timing
#include <iostream>  // std::cout, std::endl
#include <stdio.h>
#include <vector>
#include <iostream>
#include <cmath>

#include "cuda_common.h"

namespace csc485b {
namespace a1      {
namespace gpu     {

/**
 * The CPU baseline benefits from warm caches because the data was generated on
 * the CPU. Run the data through the GPU once with some arbitrary logic to
 * ensure that the GPU cache is warm too and the comparison is more fair.
 */
__global__
void warm_the_gpu( element_t * data, std::size_t invert_at_pos, std::size_t num_elements )
{
    int const th_id = blockIdx.x * blockDim.x + threadIdx.x;

    // We know this will never be true, because of the data generator logic,
    // but I doubt that the compiler will figure it out. Thus every element
    // should be read, but none of them should be modified.
    if( th_id < num_elements && data[ th_id ] > num_elements * 100 )
    {
        ++data[ th_id ]; // should not be possible.
    }
}

template<typename T>
__device__
void swap(unsigned int i, unsigned int j, T* input) {
    T temp_num = input[i];
    input[i] = input[j];
    input[j] = temp_num;
}

template <typename T>
__device__
void binary_bitonic_sort(unsigned const int thread_id, T* input, int n) {


    int bit_shift = 0;


    // the steps to progressively sort the array
    for (int step = 1; step < n; step = step << 1) {

        // creates a bit mask that will determine if a thread should be sorted in ascending or descending order
        // if the result of the bit mask is 0 or 1, it should be sorted in ascending order
        // if the result of the bit mask is 2 or 3, it should be sorted in descending order
        int order_bit = (thread_id & (0b11 << bit_shift)) >> bit_shift;
        bit_shift += 1;


        // sorts within the steps
        for (int sub_step = step; sub_step > 0; sub_step = sub_step >> 1) {

            int paired_thread_id = thread_id ^ sub_step;

            if (thread_id > paired_thread_id) {

                // if the thread should be sorted in ascending order and it is less than its pair thread
                if (input[thread_id] < input[paired_thread_id] && (order_bit == 0b01 || order_bit == 0b00)) {
                    swap(thread_id, paired_thread_id, input);
                }
                // if the thread should be sorted in descending order and it is greater than its pair thread
                else if (input[thread_id] > input[paired_thread_id] && (order_bit == 0b11 || order_bit == 0b10)) {
                    swap(thread_id, paired_thread_id, input);

                }
            }
            __syncthreads();

        }
    }

    
}

/**
 * Your solution. Should match the CPU output.
 */
__global__
void opposing_sort( element_t * data, std::size_t invert_at_pos, std::size_t num_elements )
{
    const int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    
    // GLOBAL IMPLEMENTATION
    binary_bitonic_sort(thread_id, data, num_elements);
    if (thread_id >= invert_at_pos) {
        binary_bitonic_sort(thread_id, data, num_elements - invert_at_pos); //Global memory
    }
    /*
    // SHARED IMPLEMENTATION
    extern __shared__ element_t smem[];
    smem[thread_id] = data[thread_id];
    __syncthreads();

    binary_bitonic_sort(thread_id, smem, num_elements); 

    if (thread_id >= invert_at_pos) {
        binary_bitonic_sort(thread_id, smem, num_elements - invert_at_pos);
    }
    __syncthreads();
    data[thread_id] = smem[thread_id];
    */
}

/**
 * Performs all the logic of allocating device vectors and copying host/input
 * vectors to the device. Times the opposing_sort() kernel with wall time,
 * but excludes set up and tear down costs such as mallocs, frees, and memcpies.
 */
void run_gpu_soln( std::vector< element_t > data, std::size_t switch_at, std::size_t n )
{
    // Kernel launch configurations. Feel free to change these.
    // This is set to maximise the size of a thread block on a T4, but it hasn't
    // been tuned. It's not known if this is optimal.
    std::size_t const threads_per_block = 1024;
    std::size_t const num_blocks =  ( n + threads_per_block - 1 ) / threads_per_block;

    // Allocate arrays on the device/GPU
    element_t * d_data;
    cudaMalloc( (void**) & d_data, sizeof( element_t ) * n );
    CHECK_ERROR("Allocating input array on device");

    // Copy the input from the host to the device/GPU
    cudaMemcpy( d_data, data.data(), sizeof( element_t ) * n, cudaMemcpyHostToDevice );
    CHECK_ERROR("Copying input array to device");

    // Warm the cache on the GPU for a more fair comparison
    warm_the_gpu<<< num_blocks, threads_per_block>>>( d_data, switch_at, n );

    // Time the execution of the kernel that you implemented
    auto const kernel_start = std::chrono::high_resolution_clock::now();
    opposing_sort<<< num_blocks, threads_per_block, threads_per_block*sizeof(element_t) >>>(d_data, switch_at, n);
    auto const kernel_end = std::chrono::high_resolution_clock::now();
    CHECK_ERROR("Executing kernel on device");

    // After the timer ends, copy the result back, free the device vector,
    // and echo out the timings and the results.
    cudaMemcpy( data.data(), d_data, sizeof( element_t ) * n, cudaMemcpyDeviceToHost );
    CHECK_ERROR("Transferring result back to host");
    cudaFree( d_data );
    CHECK_ERROR("Freeing device memory");

    std::cout << "GPU Solution time: "
              << std::chrono::duration_cast<std::chrono::nanoseconds>(kernel_end - kernel_start).count()
              << " ns" << std::endl;

    for( auto const x : data ) std::cout << x << " "; std::cout << std::endl;
}

} // namespace gpu
} // namespace a1
} // namespace csc485b