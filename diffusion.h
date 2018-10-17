/*
 * diffusion.h
 *
 *  Created on: 28.09.2018
 *      Author: Karl Royen
 */

#ifndef DIFFUSION_H_
#define DIFFUSION_H_


#include "poisson.cuh"

typedef struct _diffusion_plan {
	poisson_plan plan;
	void* dest;
} diffusion_plan;

/** Solve the diffusion equation for the passed parameter set and return the
 * result in u.
 *
 * The initial condition pointed to by the pointer u is overwritten with the
 * result of the diffusion equation.
 */
void diffusion_direct(float* u, int nx, int ny, int nz,
	double dx, double dy, double dz, double dt,
	int num_steps, int dirichlet);

/** Create a plate for solving the diffusion equation with the passed
 * parameters.
 */
diffusion_plan diffusion_create_plan(int nx, int ny, int nz,
	double dx, double dy, double dz, double dt,
	int dirichlet, int num_gpus);

/** Copy the passed pointer of the dataset to use as starting point for the
 * next time step to the GPU.*/
void diffusion_plan_copy_to_gpu(diffusion_plan p, float* src);

/** Execute one time step of the passed plan.*/
void diffusion_execute_plan(diffusion_plan p);

/** Copy the result of the last time step from GPU to the passed pointer.*/
void diffusion_plan_copy_to_host(diffusion_plan p, float* dst);

/** Free all resources allocated by the passed plan.*/
void diffusion_destroy_plan(diffusion_plan p);





/** Solve the diffusion equation for the passed parameter set and return the
 * result in u.
 *
 * The initial condition pointed to by the pointer u is overwritten with the
 * result of the diffusion equation.
 *
 * This method is only for comparison purposes. Use diffusion_direct instead!
 */
void diffusion(float* u, int nx, int ny, int nz,
	double dx, double dy, double dz, double dt,
	int numsteps, int dirichlet, int num_gpus);


#endif /* DIFFUSION_H_ */
