/*
 * main.cpp
 *
 *  Created on: 12.03.2018
 *      Author: Karl Royen
 */

#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <sstream>
#include <cmath>
#include "diffusion.h"

// uncomment to execute some functionality and speed tests that don't depend
// on any input file or parameter
//#define TEST

#define PI 3.14159265359


float* read_ic_bin(const char* path, size_t size) {
	float* data = new float[size];

    std::stringstream bin_file;
	bin_file << path;
    path=bin_file.str().c_str();

	FILE *temp_ptr = fopen(path, "rb");
	fread(data, sizeof(float), size, temp_ptr);
	fclose(temp_ptr);

	return data;
}

void save_data(const char* path, float* data, size_t size) {
	std::stringstream bin_file;
	bin_file << path;
	path = bin_file.str().c_str();
	FILE *temp_ptr = fopen(path, "wb");
	fwrite(data, sizeof(float), size, temp_ptr);

	fclose(temp_ptr);
}

#ifndef TEST
int main(int argc, const char **argv)
{
	const char *ic_name = "../../data/ic/temp32.dat";
	const char *result_file = "test32.temp.dat";
	int n = 32;
	int num_gpu = 1;
	int num_steps = 50;
	float dt = 1e-6; //0.12f/num_steps;
	int dirichlet = 0;
	if(argc > 1) {
		ic_name=argv[1];
		result_file=argv[2];
		n = atoi(argv[3]);
		num_steps = atoi(argv[4]);
		dirichlet = atoi(argv[5]);
	}
	if(argc > 6) num_gpu=atoi(argv[6]);
	if(argc > 7) dt=atof(argv[7]);

	int nx = n, ny = n, nz = n;

	//geo_mgpu();
	printf("Reading input %s of size %d\n", ic_name, nx*ny*nz*4);
	float* data = read_ic_bin(ic_name, nx*ny*nz);

#ifndef STEPWISE
	diffusion_direct(data, nx, ny, nz, 1.0/nx, 1.0/ny, 1.0/nz, dt, num_steps,
						dirichlet);
#else
	diffusion(data, nx, ny, nz, 1.0/nx, 1.0/ny, 1.0/nz, dt, num_steps,
					dirichlet, num_gpu);
#endif

	printf("Saving result to %s\n", result_file);
	save_data(result_file, data, nx*ny*nz);

	delete[] data;
	return 0;
}


#else
#include <cuda_runtime_api.h>
#include <time.h>





float fn(float x, float y, float z, float t)
{
	return cos(z*2*PI)*cos(y*2*PI)*sin(x*2*PI)*exp(-12*pow(PI,2)*t);
}

float* create_test_ic(float* data, bool dir, int n)
{
	float* res = data;
	float* it = res;
	float x0 = 0.0;
	float stepx = 1.0/n;
	if(dir){
		stepx = 1.0/(n+1);
		x0 = stepx;
	}
	for(float x = x0; x < 1.0-stepx/2.0f; x+=stepx){
		for(float y = 0.0; y < 1.0-stepx/2.0f; y+=1.0/n){
			for(float z = 0.0; z < 1.0-stepx/2.0f; z+=1.0/n){
				*(it++) = fn(x,y,z,0.0f);
			}
		}
	}
	return res;
}

void check_data(float* data, float t, bool dir, int n)
{
	double sum = 0.0;
	float* it = data;
	float max = -1000;
	float maxd = -1000;
	float x0 = 0.0;
	float stepx = 1.0/n;
	if(dir){
		stepx = 1.0/(n+1);
		x0 = stepx;
	}

	for(float x = x0; x < 1.0-stepx/2.0f; x+=stepx){
		for(float y = 0.0; y < 1.0-stepx/2.0f; y+=1.0/n){
			for(float z = 0.0; z < 1.0-stepx/2.0f; z+=1.0/n){
				float f = fn(x,y,z,t);
				if(fabs(f) > max)
					max = fabs(f);
				float err = fabs(*(it++) - f);
				if(err > maxd)
					maxd = err;
				sum += err;
			}
		}
	}
	double mean = sum / n/n/n;
	printf("Error:\n\tmean: %g\n\trel.mean: %g\n", mean, mean/max);
	printf("\trel.max: %g\n", maxd/max);
	printf("\trel.mean/last rel.mean %g\n", mean/max/4.95218e-06);
}



void test_new(int n, int num_steps, double dt, int dirichlet, int num_gpus)
{
	printf("\ndiffusion_execute_plan() %d x %d:\n", n, num_gpus);
	float* data = new float[n*n*n];
	create_test_ic(data, false, n);

	size_t freem, total, freeafter;
	cudaDeviceSynchronize();
	cudaMemGetInfo(&freem, &total);
	diffusion_plan p = diffusion_create_plan(n,n,n,1.0/n,1.0/n,1.0/n,dt,
				dirichlet, num_gpus);
	diffusion_plan_copy_to_gpu(p, data);
	cudaDeviceSynchronize();
	cudaMemGetInfo(&freeafter, &total);
	printf("Mem usage: %f blocks\n", (double)(freem-freeafter)/(n*n*n*4));



	cudaDeviceSynchronize();
	clock_t t = clock();
	for(int i=0; i<num_steps; ++i) {
		diffusion_execute_plan(p);
	}
	cudaDeviceSynchronize();
	t = clock() - t;

	diffusion_plan_copy_to_host(p, data);
	check_data(data, num_steps*dt, false, n);
	double time_taken = ((double)t)/CLOCKS_PER_SEC; // in seconds
	printf("Took %f seconds to execute \n", time_taken);
	diffusion_destroy_plan(p);
	delete[] data;
}

void test_new_diff(int n, int num_steps, double dt, int dirichlet, int num_gpus)
{
	printf("\ndiffusion() %d x %d:\n", n, num_gpus);
	float* data = new float[n*n*n];
	create_test_ic(data, false, n);

	cudaDeviceSynchronize();
	clock_t t = clock();
	diffusion(data, n, n, n, 1.0/n, 1.0/n, 1.0/n, dt, num_steps, dirichlet, num_gpus);
	cudaDeviceSynchronize();
	t = clock() - t;

	check_data(data, num_steps*dt, false, n);
	double time_taken = ((double)t)/CLOCKS_PER_SEC; // in seconds
	printf("Took %f seconds to execute \n", time_taken);
	delete[] data;
}

void test_new_solve(int n, int num_steps, double dt, int dirichlet, int num_gpus)
{
	printf("\ndiffusion_direct() %d x %d:\n", n, num_gpus);
	float* data = new float[n*n*n];
	create_test_ic(data, false, n);

	cudaDeviceSynchronize();
	clock_t t = clock();
	diffusion_direct(data, n, n, n, 1.0/n, 1.0/n, 1.0/n, dt, num_steps, dirichlet);
	cudaDeviceSynchronize();
	t = clock() - t;

	check_data(data, num_steps*dt, false, n);
	double time_taken = ((double)t)/CLOCKS_PER_SEC; // in seconds
	printf("Took %f seconds to execute \n", time_taken);
	delete[] data;
}

int main(int argc, const char **argv)
{
	int steps = 10000;
	int n = 32;

	printf("=== Periodic Tests ===\n");
	test_new_solve(n, steps, 1e-6, 0, 1);
	test_new_diff(n, steps, 1e-6, 0, 1);
	test_new(n, steps, 1e-6, 0, 1);

	printf("\n=== Dirichlet Tests ===\n");
	test_new_solve(n, steps, 1e-6, 1, 1);
	test_new_diff(n, steps, 1e-6, 1, 1);
	test_new(n, steps, 1e-6, 1, 1);

	return 0;
}
#endif
