#include <stdio.h>
#include <vector>
#include <iostream>
#include <cmath>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

template <typename T>
void print_vector(std::vector<T> vec);


template<typename T>
__device__
void swap(unsigned int i, unsigned int j, T* input) {
	// i = thread_id
	// j = paired_thread_id
	T temp_num = input[i];
	input[i] = input[j];
	input[j] = temp_num;
}



// not working correctly
template <typename T>
__device__
void handle_step_one(unsigned const int thread_id, T* output, T* input, int n) {

	if ((thread_id + 1) % 6 == 0) {

		if (input[thread_id] > input[thread_id - 2]) {
			swap(thread_id, (thread_id - 2), input);
		}

		if (input[thread_id] > input[thread_id - 1]) {
			swap(thread_id, (thread_id - 1), input);
		}
	} else if ((thread_id + 1) % 3 == 0) {

		if (input[thread_id] < input[thread_id - 2]) {
			swap(thread_id, (thread_id - 2), input);
		}
		
		if (input[thread_id] < input[thread_id - 1]) {
			swap(thread_id, (thread_id - 1), input);
		}
	}

}



template <typename T>
__device__
void three_bitanic_sort(unsigned const int thread_id, T* output, T* input, int n) {


	//demo
	for (int step = 0; step < n; step += 3) {

		for (int sub_step = step; sub_step >= 0; sub_step -= 3) {
			if (step == 0) {
				handle_step_one(thread_id, output, input, n);
			}
		}
	}


	output[thread_id] = input[thread_id];
}


template <typename T>
__device__
void binary_bitanic_sort(unsigned const int thread_id, T* output, T* input, int n) {
	
	//const int num_steps = log2(n);

	//const int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
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

		}
	}

	output[thread_id] = input[thread_id];

}


template <typename T>
__global__
void new_bitanic_sort(T* output, T* input, int n) {
	
	const int num_steps = log2f(n);

	const int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int bit_shift = 0b11;

	unsigned int bit_order = (thread_id & (bit_shift << (num_steps - 2))) >> (num_steps - 2);

	binary_bitanic_sort(thread_id, output, input, n);

	if (bit_order == 0b11) {
		binary_bitanic_sort(thread_id, output, input, n / 4);
	}
	output[thread_id] = input[thread_id];
}



void bitanic_sort_host() {
	//std::vector<int> input = { 4, 7, 1, 10, 2, 6, 3, 4 };


	// figure out how to use blocks
	std::vector<int> input;

	input = {
		88, 67, 64, 2, 82,
		58, 10, 81, 79, 81,
		23, 64, 23, 90, 91,
		15, 8, 93, 78, 8,
		10, 71, 96, 53, 4,
		53, 61, 18, 72, 72,
		38, 26
	};
	
	std::cout << "pre" << std::endl;
	print_vector(input);

	std::vector<int> output(input.size(), -1);

	int* d_input;
	int* d_output;

	cudaMalloc((void**)&d_input, sizeof(int) * input.size());
	cudaMalloc((void**)&d_output, sizeof(int)* output.size());

	cudaMemcpy(d_input, input.data(), sizeof(int) * input.size(), cudaMemcpyHostToDevice);
	cudaMemcpy(d_output, output.data(), sizeof(int) * output.size(), cudaMemcpyHostToDevice);

	new_bitanic_sort << <1, input.size() >> > (d_output, d_input, input.size());

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
