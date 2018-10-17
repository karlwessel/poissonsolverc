Implementation of a Fourier transform based Poissonsolver as well as a solver
for the Diffusion equation using that Poissonsolver.

Requirements
- FFTW3 library (package libfftw3-dev under debian)
- cuda sdk 8.0 or greater (package nvidia-cuda-toolkit under debian)


The source can be compiled with
```
nvcc -lfftw3 -lcufft -o diffusion poisson.cu diffusion.cu main.cpp
```

The command line arguments to solve the diffusion equation for a dataset with 
domain [0,1]x[0,1]x[0,1] in space for a certain 
- initial condition, 
- spacial resolution,
- number of timesteps of size 
- delta t, 
- periodic or dirichlet boundary condition and
- using certain number of gpus
are
```
./diffusion <IC-file> <result-file> <resolution> <numsteps> <BC> <numgpus> <deltat>
```

Where
- <IC-file> is the path of the file containing the discretized initial condition 
stored as a NxNxN array in c-compatible row major format of floating point 
values
- <result-file> is the destination file to write the solution of the diffusion
equation to (again in c-compatible row major format of float values)
- <resolution> is the resolution N of the initial condition from which the 
sampling distances in each direction is calculated as 1/N
- <numsteps> is the number of timesteps of size <deltat> to execute
- <BC> is the boundary condition to use 0 for all periodic and 
1 for Dirichlet in x and periodic in y and z
- <numgpus> the number of gpus to use
- <deltat> the size of the time step
