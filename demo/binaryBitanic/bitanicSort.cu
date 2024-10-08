#include <stdio.h>
#include <vector>
#include <iostream>
#include <cmath>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <limits>

template <typename T>
void print_vector(std::vector<T> vec);

template <typename T>
__global__
void bitanic_sort(T* output, T* input, int n) {
	
	//const int num_steps = log2(n);

	const int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
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

					T temp_num = input[thread_id];
					input[thread_id] = input[paired_thread_id];
					input[paired_thread_id] = temp_num;
				}
				// if the thread should be sorted in descending order and it is greater than its pair thread
				else if (input[thread_id] > input[paired_thread_id] && (order_bit == 0b11 || order_bit == 0b10)) {

					T temp_num = input[thread_id];
					input[thread_id] = input[paired_thread_id];
					input[paired_thread_id] = temp_num;
				}
			}
			__syncthreads();

		}
	}
		//__syncthreads();

	output[thread_id] = input[thread_id];

}




void bitanic_sort_host() {
	//std::vector<int> input = { 4, 7, 1, 10, 2, 6, 3, 4 };


	// figure out how to use blocks
	std::vector<int> input;
	
	for (int i = 16; i > 0; i--)
		input.push_back(i);

	/*
	int n = input.size();

	if (n > 0 && (n & (n - 1)) == 0)
		std::cout << "size is right\n";
	else {
		std::cout << "size is not right\n";
		int right_n = pow(2, std::ceil(log2(n)));
		std::cout << "correct size is: " << right_n << std::endl;
		std::cout << std::numeric_limits<int>::max() << std::endl;
		for (int i = input.size() - 1; i < right_n; i++)
			input.push_back(std::numeric_limits<int>::max());
	}
	*/
	/*
	input = {
		88, 67, 64, 2, 82,
		58, 10, 81, 79, 81,
		23, 64, 23, 90, 91,
		15, 8, 93, 78, 8,
		10, 71, 96, 53, 4,
		53, 61, 18, 72, 72,
		38, 26
	};
	*/
	std::cout << "pre" << std::endl;
	print_vector(input);

	std::vector<int> output(input.size(), -1);

	int* d_input;
	int* d_output;

	cudaMalloc((void**)&d_input, sizeof(int) * input.size());
	cudaMalloc((void**)&d_output, sizeof(int)* output.size());

	cudaMemcpy(d_input, input.data(), sizeof(int) * input.size(), cudaMemcpyHostToDevice);
	cudaMemcpy(d_output, output.data(), sizeof(int) * output.size(), cudaMemcpyHostToDevice);

	bitanic_sort << <2, input.size() / 2 >> > (d_output, d_input, input.size());

	cudaMemcpy(output.data(), d_output, sizeof(int) * input.size(), cudaMemcpyDeviceToHost);
	std::cout << "post" << std::endl;
	print_vector(output);
}

template <typename T>
void print_vector(std::vector<T> vec) {

	std::cout << "[";
	for (int i = 0; i < vec.size() - 1; i++)
		std::cout << vec[i] << ", ";

	std::cout <<vec[vec.size() - 1] << "]" << std::endl;


}

int main() {


	//std::cout << (15 & 0b11) << std::endl;
	//add_device_int();
	bitanic_sort_host();
	//std::cout << log2(8) << std::endl;

	return 0;
}
