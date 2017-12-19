#include "DNSGPU.cuh"

#include <iostream>
#include <fstream>
#include <math.h>
#include <time.h>
#include <vector>
#include <iomanip>
#include <algorithm>
#include <string>
#include <map>
#include <stdint.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/reduce.h>

#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <cuda.h>


// Define variables in global space (bad practice but we'll go with this for now)
// Enclosed with empty namespace to avoid linking errors with the other implementation of the algorithm
namespace 
{
	using namespace std;

	DNSGPU::SIMSTATES s_nextState = DNSGPU::SIMSTATES::INIT;

	// output format
	float start_clock = clock();

	ofstream f("result_gpu.txt"); // Solution Results
	
	ofstream g("convergence_gpu.txt"); // Convergence history


	// Input parameters 
	//float Re, Pr, Fr, T_L, T_0, T_amb, ni, nj, dx, dy, t, ny, nx, eps, /*beta,*/ iter, maxiter, tf, st, counter, column, u_wind, T_R, Lx, Ly;
	
	float Lx = 4.0, Ly = 5.0; // Domain dimensions
	int ni = 2; // Number of nodes per unit length in x direction
	int nj = 2; // Number of nodes per unit length in y direction
	int nx = Lx * ni; 
	int ny = Ly * nj; // Number of Nodes in each direction
	float u_wind = 1; // Reference velocity
	float st = 0.00005; // Total variance criteria
	float eps = 0.001; // Pressure convergence criteria
	float tf = 100; // Final time step
	float Pr = 0.5*(0.709 + 0.711); // Prandtl number
	float Re = 200.0; float Fr = 0.3; // Non-dimensional numbers for inflow conditions
	float dx = Lx / (nx - 1); 
	float dy = Ly / (ny - 1); // dx and dy
											//beta = 1; // Successive over relaxation factor (SOR)
	float t = 0; // Initial time step
	float T_L = 100.0; // Left wall temperature (C)
	float T_R = 50.0; // Right wall temperature (C)
	float T_amb = 25.0; // Ambient air temperature (C)
	float T_0 = 50.0; // Initial air temperature
	
	int maxiter = 1000; // Maximum iteration at each time step
	int counter = 0; // initial row for output monitoring
	int column = 1; // Column number for output display
				// Records number of clicks a step takes
	std::map<string, uint32_t> stepTimingAccumulator;

	// Output vector (goes out to visualize data)
	std::vector<float> vizOutput;

	// Host Vectors

	thrust::host_vector<float> u(nx * (ny + 1));
	thrust::host_vector<float> us(nx*(ny + 1));
	thrust::host_vector<float> uold(nx * (ny + 1));
	int wu = ny + 1;

	thrust::host_vector<float> v((nx + 1) * ny);
	thrust::host_vector<float> vs((nx + 1) * ny);
	thrust::host_vector<float> vold((nx + 1) * ny);
	int wv = ny;

	thrust::host_vector<float> p((nx + 1) * (ny + 1));
	int wp = ny + 1;



	thrust::host_vector<float> T;
	int wT = ny + 1;

	// These are initialized in the Init() function
	thrust::host_vector<float> Told;
	thrust::host_vector<float> om;
	thrust::host_vector<float> vc;
	thrust::host_vector<float> uc;
	thrust::host_vector<float> pc;
	thrust::host_vector<float> Tc;
	int wc = ny;

	thrust::device_vector<float> us_d(nx*(ny + 1));
	thrust::device_vector<float> vs_d((nx + 1) * ny);
	thrust::device_vector<float> p_d((nx + 1) * (ny + 1));
	thrust::device_vector<float> p_old((nx + 1) * (ny + 1));
	thrust::device_vector<float> p_ref((nx + 1) * (ny + 1));
	thrust::device_vector<float> abs_d((nx + 1) * (ny + 1));

#define BLOCK_SIZE 32 // Number of threads in x and y direction - Maximum Number of threads per block = 32 * 32 = 1024

	// Time step size stability criterion

	float mt1 = 0.25*pow(dx, 2.0) / (1.0 / Re); float Rer = 1.0 / Re; float mt2 = 0.25*pow(dy, 2.0) / (1.0 / Re);
	float dt;

}


void DNSGPU::RunSimulation()
{
	switch (s_nextState)
	{
	case INIT: Init(); break;
	case STEP: Step(); break;
	case FINISH: Finish(); break;
	case END: break;
	}


}

// Call this to initialize shit
void DNSGPU::Init()
{
	f.setf(ios::fixed | ios::showpoint);
	f << setprecision(5);

	g.setf(ios::fixed | ios::showpoint);
	g << setprecision(5);
	cout.setf(ios::fixed | ios::showpoint);
	cout << setprecision(5);

	T_L = T_L + 273.15; T_0 = T_0 + 273.15; T_amb = T_amb + 273.15; T_R = T_R + 273.15;// Unit conversion to (K)

	T.resize((nx + 1) * (ny + 1), T_0 / T_amb);
	Told.resize((nx + 1) * (ny + 1), 0);
	om.resize(nx * ny, 0);
	vc.resize(nx * ny, 0);
	uc.resize(nx * ny, 0);
	pc.resize(nx * ny, 0);
	Tc.resize(nx*ny, 0);

	if (mt1 > Rer)
	{
		dt = Rer;
	}
	else
	{
		dt = mt1;
	}

	if (dt > mt2)
	{
		dt = mt2;
	}
	cout << "dt = " << dt << endl;

	s_nextState = STEP;
}

__global__ void Temperature_solver(int nx, int ny, int wu, int wv, int wT, float dx, float dy, float dt, float Re, float Pr, float *u, float *v, float *Told, float *T)
{

	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;

	if (i > 0 && i < nx && j > 0 && j < ny) {

		Told[i * wT + j] = T[i * wT + j];
		T[i * wT + j] = T[i * wT + j] + dt*(-0.5*(u[i * wu + j] + u[(i - 1) * wu + j])*(1.0 / (2.0*dx)*(T[(i + 1) * wT + j] - T[(i - 1) * wT + j])) - 0.5*(v[i * wv + j] + v[i * wv + j - 1])*(1.0 / (2.0*dy)*(T[i * wT + j + 1] - T[i * wT + j - 1])) + 1 / (Re*Pr)*(1 / pow(dx, 2.0f)*(T[(i + 1) * wT + j] - 2.0*T[i * wT + j] + T[(i - 1) * wT + j]) + 1 / pow(dy, 2.0f)*(T[i * wT + j + 1] - 2 * T[i * wT + j] + T[i * wT + j - 1])));
	}
	__syncthreads();
}



__global__ void PressureSolve(float * p_d, const float * p_old, float * abs_d, const float * us_d, const float * vs_d, int p_xlength, int p_ylength, int wp, int wu, int wv, float dx, float dy, float dt)
{

	int i = threadIdx.x + blockDim.x*blockIdx.x;
	int j = threadIdx.y + blockDim.y*blockIdx.y;


	if (i > 0 && i < p_xlength && j > 0 && j < p_ylength)
	{
		//        __syncthreads();

		p_d[i * wp + j] = pow(dx, 2.0f)*pow(dy, 2.0f) / (-2.0*(pow(dx, 2.0f) + pow(dy, 2.0f)))*(-1.0 / pow(dx, 2.0f)*(p_old[(i + 1) * wp + j] + p_old[(i - 1) * wp + j] + p_old[i * wp + j + 1] + p_old[i * wp + j - 1]) + 1.0 / dt*(1.0 / dx*(us_d[i * wu + j] - us_d[(i - 1) * wu + j]) + 1.0 / dy*(vs_d[i * wv + j] - vs_d[i * wv + j - 1])));
		__syncthreads();

		abs_d[i * wp + j] = p_d[i * wp + j] - p_old[i * wp + j];

		__syncthreads();

		abs_d[i * wp + j] = abs_d[i * wp + j] * abs_d[i * wp + j];
		//__syncthreads();
	} // end if


} // end global


__global__ void PressureBC(float * p_d, float * p_ref, int nx, int ny, float dy, int wp)
{

	int i = threadIdx.x + blockDim.x*blockIdx.x;
	int j = threadIdx.y + blockDim.y*blockIdx.y;

	if (i >= 0 && i < nx + 1 && j == 0) {
		p_d[i * wp + j] = p_ref[i * wp + j + 1]; // bottom wall - Final
	}
	__syncthreads();
	if (i >= 0 && i < nx + 1 && j == ny) {
		p_d[i * wp + j] = p_ref[i * wp + j - 1]; // Upper - no flux
	}
	__syncthreads();
	if (j >= 0 && j < ny + 1 && i == 0) {
		p_d[i * wp + j] = p_ref[(i + 1) * wp + j]; // left wall - not the inlet - Final
	}
	__syncthreads();
	if (j >= 0 && j < ny + 1 && i == nx && j*dy < 2.0) {
		p_d[i * wp + j] = p_ref[(i - 1) * wp + j]; // right wall - not the outlet - Final

												   // printf("POSITIVE ");
	}
	__syncthreads();
	if (j >= 0 && j < ny + 1 && i == nx && j*dy >= 2.0) {
		p_d[i * wp + j] = -p_ref[(i - 1) * wp + j]; // pressure outlet - static pressure is zero - Final
													// printf("NEGATIVE ");    

	}
	//__syncthreads();

}

void DNSGPU::Step()
{
	cudaDeviceSynchronize();
	// dt is defined statically 
	if(t<tf)
	{
		int stepi1_start = clock();
		//........................................................................................
		// Step i1 - it can be parallelized 
		// boundary conditions for u velocity

		for (int i = 0; i < nx; i++)
		{
			for (int j = 0; j < ny + 1; j++)
			{
				if (i == 0 && j > 0 && j < ny)
				{
					if (j*dy < 2.0)
					{
						u[i * wu + j] = 0; // left wall - Final
					}
					else
					{
						u[i * wu + j] = u_wind; // left inlet - Final
					}
				}
				else if (i == nx - 1 && j>0 && j < ny)
				{
					if (j*dy < 2.0)
					{
						u[i * wu + j] = 0; // Right wall has 0 horizontal velocity - Final
					}
					else
					{
						u[i * wu + j] = u[(i - 1) * wu + j]; // right outlet - no velocity change
					}
				}
				else if (j == 0)
				{
					u[i * wu + j] = -u[i * wu + j + 1]; // bottom ghost - Final
				}
				else if (j == ny)
				{
					u[i * wu + j] = u[i * wu + j - 1]; // upper ghost - Final
				}
			} // end for j
		} // end for i
		int stepi1_end = clock();
		stepTimingAccumulator["Step i1 - Set Horizontal Velocity Boundary Conditions"] += stepi1_end - stepi1_start;
		//...............................................................................................


		//.........................................................................................
		// Step i2 - it can be parallelized
		// boundary conditions for v velocity
		int stepi2_start = clock();

		for (int i = 0; i < nx + 1; i++)
		{
			for (int j = 0; j < ny; j++)
			{
				if (j == 0 && i > 0 && i < nx)
				{
					v[i * wv + j] = 0; // bottom wall - Final
				}
				else if (j == ny - 1 && i > 0 && i < nx)
				{
					v[i * wv + j] = v[i * wv + j - 1]; // upper wall - Final
				}
				else if (i == 0)
				{
					v[i * wv + j] = -v[(i + 1) * wv + j]; // left ghost (Left Wall and inlet has 0 vertical velocity) - Final
				}
				else if (i == nx)
				{
					if (j*dy < 2.0)
					{
						v[i * wv + j] = -v[(i - 1) * wv + j]; // right ghost (Right wall has 0 vertical velocity) - Final
					}
					else
					{
						v[i * wv + j] = v[(i - 1) * wv + j]; // right outlet - no velocity gradient
					}
				}
			} // end for j
		} // end for I
		int stepi2_end = clock();
		stepTimingAccumulator["Step i2 - Set Vertical Velocity Boundary Conditions"] += stepi2_end - stepi2_start;
		//...............................................................................................

		//...............................................................................................
		int step1_start = clock();
		//.........................................................................................
		// Step 1 - it can be parallelized - Solve for intermediate velocity values

		// u - us - vh - a 

		for (int i = 1; i < nx - 1; i++)
		{
			for (int j = 1; j < ny; j++)
			{
				float vh = 1.0 / 4.0*(v[i * wv + j] + v[(i + 1) * wv + j] + v[i * wv + j - 1] + v[(i + 1) * wv + j - 1]); // v hat
				float a = u[i * wu + j] * 1.0 / (2.0*dx)*(u[(i + 1) * wu + j] - u[(i - 1) * wu + j]) + vh*1.0 / (2.0*dy)*(u[i * wu + j + 1] - u[i * wu + j - 1]); // a
				us[i * wu + j] = dt / Re*(1.0 / pow(dx, 2.0)*(u[(i + 1) * wu + j] - 2.0*u[i * wu + j] + u[(i - 1) * wu + j]) + 1.0 / pow(dy, 2.0)*(u[i * wu + j + 1] - 2.0*u[i * wu + j] + u[i * wu + j - 1])) - a*dt + u[i * wu + j]; // u star
			} // end for j
		} // end for i

		  //..........................................................................................
		  // Step 1 - it can be parallelized
		  // v - vs - uh - b
		for (int i = 1; i < nx; i++)
		{
			for (int j = 1; j < ny - 1; j++)
			{
				float uh = 1.0 / 4.0*(u[i * wu + j] + u[i * wu + j + 1] + u[(i - 1) * wu + j] + u[(i - 1) * wu + j + 1]);
				float b = uh*1.0 / (2.0*dx)*(v[(i + 1) * wv + j] - v[(i - 1) * wv + j]) + v[i * wv + j] * 1.0 / (2.0*dy)*(v[i * wv + j + 1] - v[i * wv + j - 1]); // b
				vs[i * wv + j] = dt / Re*(1.0 / pow(dx, 2.0)*(v[(i + 1) * wv + j] - 2.0*v[i * wv + j] + v[(i - 1) * wv + j]) + 1.0 / pow(dy, 2.0)*(v[i * wv + j + 1] - 2.0*v[i * wv + j] + v[i * wv + j - 1])) + dt / pow(Fr, 2.0)*(0.5*(T[i * wT + j] + T[i * wT + j - 1]) - 1) / (0.5*(T[i * wT + j] + T[i * wT + j - 1])) - b*dt + v[i * wv + j]; // v 
			} // end for j
		} // end for i

		  //...........................................................................................
		  // vs and us on Boundary conditions

		for (int i = 0; i < nx; i++)
		{
			us[i * wu + 0] = -us[i * wu + 1]; // bottom ghost - Final
		} // end for j

		  //...........................................................................................
		for (int j = 0; j < ny + 1; j++)
		{
			if (j*dy < 2.0)
			{
				us[0 * wu + j] = 0; // left wall - FInal
				us[(nx - 1) * wu + j] = 0; // right wall - Final
			}
			else
			{
				us[0 * wu + j] = u_wind; // left inlet - Final
			}
		}
		//...........................................................................................

		for (int j = 0; j < ny; j++)
		{
			vs[0 * wv + j] = -vs[1 * wv + j]; // left ghost (Both wall and inlet have 0 vs) - Final
			if (j*dy < 2.0)
			{
				vs[nx * wv + j] = -vs[(nx - 1) * wv + j]; // right ghost (Only the right wall - Final
			}
			else
			{
				vs[nx * wv + j] = vs[(nx - 1) * wv + j]; // right outlet - no flux
			}
		}
		//............................................................................................

		for (int i = 0; i < nx + 1; i++)
		{
			vs[i * wv + 0] = 0; // Bottom wall - Final
		} // end for i
		  //............................................................................................

		int step1_end = clock();
		stepTimingAccumulator["Step 1 - Solve for intermediate velocities"] += step1_end - step1_start;

		//...............................................................................................
		// Step 2 - It can be parallelized 
		// This is the most expensive part of the code
		// Poisson equation for pressure
		int step2_start = clock();
		// Cuda set up
		int p_xlength = nx;
		int p_ylength = ny;

		float *ptr_us = thrust::raw_pointer_cast(&us_d[0]);
		float *ptr_vs = thrust::raw_pointer_cast(&vs_d[0]);
		float *ptr_p = thrust::raw_pointer_cast(&p_d[0]);
		float *ptr_p_old = thrust::raw_pointer_cast(&p_old[0]);
		float *ptr_abs = thrust::raw_pointer_cast(&abs_d[0]);
		float *ptr_p_ref = thrust::raw_pointer_cast(&p_ref[0]);

		float error = 1.0; int iter = 0;
		// float diffp = 0;
		us_d = us;
		vs_d = vs;

		// Begin Jacobi loop
		while (error > eps && iter < maxiter) {
			p_old = p_d;

			// Jacobi pressure solver
			PressureSolve <<< dim3((ny + 1) / BLOCK_SIZE + 1, (nx + 1) / BLOCK_SIZE + 1, 1), dim3(BLOCK_SIZE, BLOCK_SIZE, 1) >> >(ptr_p, ptr_p_old, ptr_abs, ptr_us, ptr_vs, p_xlength, p_ylength, wp, wu, wv, dx, dy, dt);
			cudaDeviceSynchronize();

			p_ref = p_d;

			error = thrust::reduce(abs_d.begin(), abs_d.end());

			PressureBC <<< dim3((ny + 1) / BLOCK_SIZE + 1, (nx + 1) / BLOCK_SIZE + 1, 1), dim3(BLOCK_SIZE, BLOCK_SIZE, 1) >> >(ptr_p, ptr_p_ref, nx, ny, dy, wp);

			cudaDeviceSynchronize();

			error = pow(error, 0.5);
			iter = iter + 1;

		} // end while eps

		p = p_d;

		int step2_end = clock();
		stepTimingAccumulator["Step 2 - Solve for pressure until tolerance or max iterations"] += step2_end - step2_start;
		//...............................................................................................

		//.................................................................................................
		// Step 3 - It can be parallelized 
		// velocity update - projection method
		int step3_start = clock();

		// u

		for (int i = 1; i < nx - 1; i++)
		{
			for (int j = 1; j < ny; j++)
			{
				uold[i * wu + j] = u[i * wu + j];
				u[i * wu + j] = us[i * wu + j] - dt / dx*(p[(i + 1) * wp + j] - p[i * wp + j]);
			} // end for j
		} // end for i
		  //................................................

		  // v

		for (int i = 1; i < nx; i++)
		{
			for (int j = 1; j < ny - 1; j++)
			{
				vold[i * wv + j] = v[i * wv + j];
				v[i * wv + j] = vs[i * wv + j] - dt / dy*(p[i * wp + j + 1] - p[i * wp + j]);
			} // end for j
		} // end for i
		int step3_end = clock();
		stepTimingAccumulator["Step 3 - Velocity Update"] += step3_end - step3_start;
		//...............................................................................................

		//...............................................................................................
		// Step 4 - It can be parallelized
		// Solving for temperature
		int step4_start = clock();

		thrust::device_vector<float> d_T = T;
		thrust::device_vector<float> d_Told = Told;
		thrust::device_vector<float> d_u = u;
		thrust::device_vector<float> d_v = v;

		int gridsize_x = nx / BLOCK_SIZE + 1;
		int gridsize_y = ny / BLOCK_SIZE + 1;

		dim3 dimgrid(gridsize_x, gridsize_y, 1); // The grid has #gridsize blocks in x and 1 block in y and 1 block in z direction
		dim3 dimblock(BLOCK_SIZE, BLOCK_SIZE, 1);

		float *ptr_u = thrust::raw_pointer_cast(&d_u[0]);
		float *ptr_v = thrust::raw_pointer_cast(&d_v[0]);
		float *ptr_T = thrust::raw_pointer_cast(&d_T[0]);
		float *ptr_Told = thrust::raw_pointer_cast(&d_Told[0]);

		Temperature_solver << <dimgrid, dimblock >> >(nx, ny, wu, wv, wT, dx, dy, dt, Re, Pr, ptr_u, ptr_v, ptr_Told, ptr_T);

		thrust::copy(d_Told.begin(), d_Told.end(), Told.begin());
		thrust::copy(d_T.begin(), d_T.end(), T.begin());

		int step4_end = clock();
		stepTimingAccumulator["Step 4 - Solving for temperature"] += step4_end - step4_start;
		//................................................................................................

		//...............................................................................................
		// Step i3 - Initializing boundary conditions for temperature 
		// boundary conditions for Temperature
		int stepi3_start = clock();

		for (int i = 0; i < nx + 1; i++)
		{
			for (int j = 0; j < ny + 1; j++)
			{
				if (j == 0)
				{
					T[i * wT + j] = T[i * wT + j + 1]; // bottom wall - Insulated - no flux - Final
				}
				else if (j == ny)
				{
					T[i * wT + j] = 2.0*(T_0) / T_amb - T[i * wT + j - 1]; // upper boundary - lid with ambient temperature (as air) - Final
				}
				else if (i == 0)
				{
					if (j*dy < 2.0)
					{
						T[i * wT + j] = 2.0*T_L / T_amb - T[(i + 1) * wT + j]; // left wall at T_L - Constant Temperature - Final
					}
					else
					{
						T[i * wT + j] = 2.0*T_0 / T_amb - T[(i + 1) * wT + j]; // left inlet at T_0 (initial temperature) - Final
					}
				}
				else if (i == nx)
				{
					if (j*dy < 2.0)
					{
						T[i * wT + j] = 2.0*T_R / T_amb - T[(i - 1) * wT + j]; // right wall at T_R - Final
					}
				}
			} // end for j
		} // end for i
		int stepi3_end = clock();
		stepTimingAccumulator["Step i3 - Initializing boundary conditions for temperature"] += stepi3_end - stepi3_start;
		//...............................................................................................

		//...............................................................................................
		// Step 5 - Checking if solution reached steady state
		// Checking the steady state condition
		int step5_start = clock();

		float TV, diffv; TV = 0;
		for (int i = 1; i < nx - 1; i++)
		{
			for (int j = 1; j < ny - 2; j++)
			{
				diffv = v[i * wv + j] - vold[i * wv + j];
				TV = TV + pow(pow(diffv, 2), 0.5);
			} // end for i
		} // end for j

		TV = TV / ((nx - 1)*(ny - 2));

		if (TV < st && error < eps)
		{
			cout << "Steady state time = " << t << " (s) " << endl;
			s_nextState = FINISH;
			return;
		}
		counter = counter + 1;
		if (fmod(counter, 10) == 0 || counter == 1)
		{
			//cout << "" << endl;
			//cout << "Column" << setw(30) << "time(s)" << setw(30) << "Iterations on Pressure" << setw(30) << "Pressure Residual" << setw(30) << "Total Variance" << endl;
		} // end if
		int step5_end = clock();
		stepTimingAccumulator["Step 5 - Check for steady state"] += step5_end - step5_start;
		//...............................................................................................


		//cout << column << setw(30) << t << setw(30) << iter << setw(30) << error << setw(30) << TV << endl;
		g << column << setw(30) << t << setw(30) << iter << setw(30) << error << setw(30) << TV << endl;
		t = t + dt;
		column = column + 1;
	}
	else
	{
		s_nextState = FINISH;
	}
}

void DNSGPU::Finish()
{
	// Step 6
	// Co-locate the staggered grid points 
	int step6_start = clock();
	for (int i = 0; i < nx; i++)
	{
		for (int j = 0; j < ny; j++)
		{
			vc[i * wc + j] = 1.0 / 2.0*(v[(i + 1) * wv + j] + v[i * wv + j]);
			pc[i * wc + j] = 1.0 / 4.0*(p[i * wp + j] + p[(i + 1) * wp + j] + p[i * wp + j + 1] + p[(i + 1) * wp + j + 1]);
			uc[i * wc + j] = 1.0 / 2.0*(u[i*wu + j] + u[i * wu + j + 1]);
			om[i * wc + j] = 1.0 / dx*(v[(i + 1) * wv + j] - v[i * wv + j]) - 1.0 / dy*(u[i * wu + j + 1] - u[i * wu + j]);
			Tc[i * wc + j] = 1.0 / 4.0*(T[i * wT + j] + T[(i + 1) * wT + j] + T[i * wT + j + 1] + T[(i + 1) * wT + j + 1]);
		} // end for j
	} // end for i
	  //........................................................................................................
	int step6_end = clock();
	stepTimingAccumulator["Step 6 - Co-locate staggered grid points"] += step6_end - step6_start;

	// Steady state results

	for (int j = 0; j < ny; j++)
	{
		for (int i = 0; i < nx; i++)
		{
			f << setw(15) << t - dt << setw(15) << i*dx << setw(15) << j*dy << setw(15) << uc[i * wc + j] << setw(15) << vc[i * wc + j] << setw(15) << pc[i * wc + j] << setw(15) << Tc[i * ny + j] * T_amb - 273.15 << setw(15) << om[i * wc + j] << endl;
		} // end for i
	} // end for j
	  //.........................................................................................................

	float end_clock = clock();
	cout << "CPU time = " << (end_clock - start_clock) / CLOCKS_PER_SEC << " (s)" << endl;
	//cout << "Re = " << Re << endl;
	//cout << "Fr = " << Fr << endl;

	for (auto it = stepTimingAccumulator.begin(); it != stepTimingAccumulator.end(); it++)
	{
		float seconds = (float)it->second / CLOCKS_PER_SEC;
		std::cout << it->first << "\t" << seconds << endl;
	}

	s_nextState = END;
}
DNSGPU::DNSGPU()
{
}


DNSGPU::~DNSGPU()
{
}


std::vector<float>& DNSGPU::getPressure()
{
	vizOutput.resize(p.size());
	for (int i = 0; i != p.size(); ++i)
		vizOutput[i] = p[i];

	return vizOutput;
}
float DNSGPU::getPressureWidth()
{
	return wp;
}

std::vector<float>& DNSGPU::getTemperature()
{
	vizOutput.resize(T.size());

	for (int i = 0; i != T.size(); ++i)
		vizOutput[i] = T[i];

	return vizOutput;
}
float DNSGPU::getTemperatureWidth()
{
	return wT;
}

std::vector<float>& DNSGPU::getU()
{
	vizOutput.resize(u.size());

	for (int i = 0; i != u.size(); ++i)
		vizOutput[i] = u[i];

	return vizOutput;
}
float DNSGPU::getUWidth()
{
	return wu;
}
