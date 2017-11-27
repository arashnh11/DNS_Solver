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

using namespace std;
int main()
{
    float start_clock = clock();
    ofstream f("result_cpu.txt"); // Solution Results
    f.setf(ios::fixed | ios::showpoint);
    f << setprecision(5);

    ofstream g("convergence_cpu.txt"); // Convergence history
    g.setf(ios::fixed | ios::showpoint);
    g << setprecision(5);
    cout.setf(ios::fixed | ios::showpoint);
    cout << setprecision(5);

    float T_B, Re, Pr, Fr, T_L, T_0, T_amb, h, dx, dy, t, ny, nx, dt, error, eps, abs, beta, iter, maxiter, tf, st, pold, counter, column, u_wind, T_R, Lx, Ly, viscosity;

    // Input parameters 
    Lx = 2 * 2.0; Ly = 5.0; // Domain dimensions
    nx = Lx * 2.0; ny = Ly * 2.0; // Grid size
    u_wind = 1; // Reference velocity
    viscosity = 0.5*(16.97 + 18.90)*pow(10.0, -6.0); // Fluid viscosity
    st = 0.00005; // Total variance criteria
    eps = 0.001; // Pressure convergence criteria
    tf = 1.0; // Final time step
    Pr = 0.5*(0.709 + 0.711); // Prandtl number
    Re = 30.0; Fr = 0.3; // Non-dimensional numbers for inflow conditions
    dx = Lx / (nx - 1); dy = Ly / (ny - 1); // dx and dy
    beta = 1.4; // Successive over relaxation factor (SOR)
    t = 0; // Initial time step
    T_L = 100.0; // Left wall temperature (C)
    T_R = 50.0; // Right wall temperature (C)
    T_amb = 25.0; // Ambient air temperature (C)
    T_0 = 50.0; // Initial air temperature
    T_L = T_L + 273.15; T_0 = T_0 + 273.15; T_amb = T_amb + 273.15; T_R = T_R + 273.15;// Unit conversion to (K)
    maxiter = 100; // Maximum iteration at each time step
    counter = 0; // initial row for output monitoring
    column = 1; // Column number for output display

    // Records number of clicks a step takes
    std::map<string, uint32_t> stepTimingAccumulator;

    // Vectors

    vector<vector<float> > u(nx, vector<float>(ny + 1));
    vector<vector<float> > us(nx, vector<float>(ny + 1));
    vector<vector<float> > uold(nx, vector<float>(ny + 1));

    vector<vector<float> > v(nx + 1, vector<float>(ny));
    vector<vector<float> > vs(nx + 1, vector<float>(ny));
    vector<vector<float> > vold(nx + 1, vector<float>(ny));

    vector<vector<float> > p(nx + 1, vector<float>(ny + 1));
    vector<vector<float> > T(nx + 1, vector<float>(ny + 1));
    vector<vector<float> > Told(nx + 1, vector<float>(ny + 1));

    vector<vector<float> > sai(nx, vector<float>(ny));
    vector<vector<float> > om(nx, vector<float>(ny));
    vector<vector<float> > vc(nx, vector<float>(ny));
    vector<vector<float> > uc(nx, vector<float>(ny));

    vector<vector<float> > pc(nx, vector<float>(ny));
    vector<vector<float> > Tc(nx, vector<float>(ny));

    // Time step size stability criterion

    float mt1 = 0.25*pow(dx, 2.0) / (1.0 / Re); float Rer = 1.0 / Re; float mt2 = 0.25*pow(dy, 2.0) / (1.0 / Re);

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

    
    //......................................................................................
    // Step 0 - It can be parallelized
    // Initializing the flow variable (Temperature)  
    // Boundary conditions for T (Initialization)
    int step0_start = clock();
    for (int i = 0; i < nx + 1; i++)
    {
        for (int j = 0; j < ny + 1; j++)
        {
            T[i][j] = T_0 / T_amb;
        } // end for j
    } // end for i
    //......................................................................................
    int step0_end = clock();
    stepTimingAccumulator["Step 0, Initializing Temperature"] += step0_end - step0_start;
    //......................................................................................

    // Marching in Time - Outermost loop

    while (t <= tf)
    {

        iter = 0;

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
                        u[i][j] = 0; // left wall - Final
                    }
                    else
                    {
                        u[i][j] = u_wind; // left inlet - Final
                    }
                }
                else if (i == nx - 1 && j>0 && j < ny)
                {
                    if (j*dy < 2.0)
                    {
                        u[i][j] = 0; // Right wall has 0 horizontal velocity - Final
                    }
                    else
                    {
                        u[i][j] = u[i - 1][j]; // right outlet - no velocity change
                    }
                }
                else if (j == 0)
                {
                    u[i][j] = -u[i][j + 1]; // bottom ghost - Final
                }
                else if (j == ny)
                {
                    u[i][j] = u[i][j - 1]; // upper ghost - Final
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
                    v[i][j] = 0; // bottom wall - Final
                }
                else if (j == ny - 1 && i > 0 && i < nx)
                {
                    v[i][j] = v[i][j - 1]; // upper wall - Final
                }
                else if (i == 0)
                {
                    v[i][j] = -v[i + 1][j]; // left ghost (Left Wall and inlet has 0 vertical velocity) - Final
                }
                else if (i == nx)
                {
                    if (j*dy < 2.0)
                    {
                        v[i][j] = -v[i - 1][j]; // right ghost (Right wall has 0 vertical velocity) - Final
                    }
                    else
                    {
                        v[i][j] = v[i - 1][j]; // right outlet - no velocity gradient
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
                float vh = 1.0 / 4.0*(v[i][j] + v[i + 1][j] + v[i][j - 1] + v[i + 1][j - 1]); // v hat
                float a = u[i][j] * 1.0 / (2.0*dx)*(u[i + 1][j] - u[i - 1][j]) + vh*1.0 / (2.0*dy)*(u[i][j + 1] - u[i][j - 1]); // a
                us[i][j] = dt / Re*(1.0 / pow(dx, 2.0)*(u[i + 1][j] - 2.0*u[i][j] + u[i - 1][j]) + 1.0 / pow(dy, 2.0)*(u[i][j + 1] - 2.0*u[i][j] + u[i][j - 1])) - a*dt + u[i][j]; // u star
            } // end for j
        } // end for i

        //..........................................................................................
        // Step 1 - it can be parallelized
        // v - vs - uh - b
        for (int i = 1; i < nx; i++)
        {
            for (int j = 1; j < ny - 1; j++)
            {
                float uh = 1.0 / 4.0*(u[i][j] + u[i][j + 1] + u[i - 1][j] + u[i - 1][j + 1]);
                float b = uh*1.0 / (2.0*dx)*(v[i + 1][j] - v[i - 1][j]) + v[i][j] * 1.0 / (2.0*dy)*(v[i][j + 1] - v[i][j - 1]); // b
                vs[i][j] = dt / Re*(1.0 / pow(dx, 2.0)*(v[i + 1][j] - 2.0*v[i][j] + v[i - 1][j]) + 1.0 / pow(dy, 2.0)*(v[i][j + 1] - 2.0*v[i][j] + v[i][j - 1])) + dt / pow(Fr, 2.0)*(0.5*(T[i][j] + T[i][j - 1]) - 1) / (0.5*(T[i][j] + T[i][j - 1])) - b*dt + v[i][j]; // v 
            } // end for j
        } // end for i

        //...........................................................................................
        // vs and us on Boundary conditions

        for (int i = 0; i < nx; i++)
        {
            us[i][0] = -us[i][1]; // bottom ghost - Final
        } // end for j

        //...........................................................................................
        for (int j = 0; j < ny + 1; j++)
        {
            if (j*dy < 2.0)
            {
                us[0][j] = 0; // left wall - FInal
                us[nx - 1][j] = 0; // right wall - Final
            }
            else
            {
                us[0][j] = u_wind; // left inlet - Final
            }
        }
        //...........................................................................................

        for (int j = 0; j < ny; j++)
        {
            vs[0][j] = -vs[1][j]; // left ghost (Both wall and inlet have 0 vs) - Final
            if (j*dy < 2.0)
            {
                vs[nx][j] = -vs[nx - 1][j]; // right ghost (Only the right wall - Final
            }
            else
            {
                vs[nx][j] = vs[nx - 1][j]; // right outlet - no flux
            }
        }
        //............................................................................................

        for (int i = 0; i < nx + 1; i++)
        {
            vs[i][0] = 0; // Bottom wall - Final
        } // end for i
        //............................................................................................

        int step1_end = clock();
        stepTimingAccumulator["Step 1 - Solve for intermediate velocities"] += step1_end - step1_start;

        //...............................................................................................
        // Step 2 - It can be parallelized 
        // This is the most expensive part of the code
        // Poisson equation for pressure
        int step2_start = clock();

        float error = 1; iter = 0;

        // Solve for pressure iteratively until it converges - Using Gauss Seidel SOR 

// Cuda set up
float *	p_h, us_h, vs_h;
float error;
p_h = new float[(nx+1)*(ny+1)];
us_h = new float[nx*(ny+1)];
vs_h = new float[(nx+1)*ny];

	
	for(int i = 0; i < nx; ++i)
	  for(int j=0; j < ny+1; ++j)
		us_h[i*wu + j] = us[i][j];	
	
	
	for(int i = 0; i < nx+1; ++i)
	  for(int j=0; j < ny; ++j)
		vs_h[i*wv + j] = vs[i][j];	
	
	
float * p_d, u_d, v_d, error_d;
cudaMalloc(&p_d, (nx+1)*(ny+1)*sizeof(float));
cudaMalloc(&u_v, (nx)*(ny+1)*sizeof(float));
cudaMalloc(&v_d, (nx+1)*(ny)*sizeof(float));
cudaMalloc(&error, sizeof(float));


cudaMemcpy(u_d, u_h, sizeof(float)*(nx)*(ny+1), cudaMemcpyHostToDevice);
cudaMemcpy(v_d, v_h, sizeof(float)*(nx+1)*(ny), cudaMemcpyHostToDevice);


        while (error > eps)
        {
            error = 0;

	for(int i = 0; i < nx+1; ++i)
	  for(int j=0; j < ny+1; ++j)
		p_h[i*wp + j] = p[i][j];	
	
cudaMemcpy(p_d, p_h, sizeof(float)*(nx+1)*(ny+1), cudaMemcpyHostToDevice);
__global PressureSolve(float * p_d, const float * u_d, const float * v_d, float* error, int wp, int wu, int wv)
{
	int i = threadIdx.x + blockDim.x*blockIdx.x;
	int j = threadIdx.y + blockDim.y*blockIdx.y;

float 	pold = p[i * wp + j];
                    p[i * wp + j] = beta*pow(dx, 2.0)*pow(dy, 2.0) / (-2.0*(pow(dx, 2.0) + pow(dy, 2.0)))*(-1.0 / pow(dx, 2.0)*(p[(i + 1) * wp + j] + p[(i - 1) * wp + j] + p[i * wp + j + 1] + p[i * wp + j - 1]) + 1.0 / dt*(1.0 / dx*(u_d[i * wu + j] - u_d[(i - 1) * wu + j]) + 1.0 / dy*(v_d[i * wv + j] - v_d[i * wv + j - 1]))) + (1.0 - beta)*p[i * wp + j];
                    abs = pow((p[i * wp + j] - pold), 2.0);
                    atomicAdd(error,abs);
}
int blockSize = 32;

PressureSolve<<< dim3( (ny+1)/blockSize + 1, (nx+1)/blockSize + 1) , dim3(blockSize,blockSize)>>>(p_d, u_d, v_d, error, wp, wu, wv);

cudaMemcpy(p_h, p_d, sizeof(float)*(nx+1)*(ny+1), cudaDeviceToHost);
cudaMemcpy(&error, error_d, sizeof(float), cudaDeviceToHost);


	for(int i = 0; i < nx+1; ++i)
	  for(int j=0; j < ny+1; ++j)
		p[i][j] = p_h[i*wp + j];	

/*
            //............................................................................................
            for (int i = 1; i < nx; i++)
            {
                for (int j = 1; j < ny; j++)
                {
                    pold = p[i][j];
                    p[i][j] = beta*pow(dx, 2.0)*pow(dy, 2.0) / (-2.0*(pow(dx, 2.0) + pow(dy, 2.0)))*(-1.0 / pow(dx, 2.0)*(p[i + 1][j] + p[i - 1][j] + p[i][j + 1] + p[i][j - 1]) + 1.0 / dt*(1.0 / dx*(us[i][j] - us[i - 1][j]) + 1.0 / dy*(vs[i][j] - vs[i][j - 1]))) + (1.0 - beta)*p[i][j];
                    abs = pow((p[i][j] - pold), 2.0);
                    error = error + abs;
                } // end for j
            } // end for i
            //............................................................................................
            // boundary conditions for pressure
*/
            for (int i = 0; i < nx + 1; i++)
            {
                for (int j = 0; j < ny + 1; j++)
                {
                    if (j == 0)
                    {
                        p[i][j] = p[i][j + 1]; // bottom wall - Final
                    }
                    else if (j == ny)
                    {
                        p[i][j] = p[i][j - 1]; // Upper - no flux
                    }
                    else if (i == 0)
                    {
                        if (j*dy < 2.0)
                        {
                            p[i][j] = p[i + 1][j]; // left wall - not the inlet - Final
                        }
                        else
                        {
                            p[i][j] = p[i + 1][j];
                        }
                    }
                    else if (i == nx)
                    {
                        if (j*dy < 2.0)
                        {
                            p[i][j] = p[i - 1][j]; // right wall - not the outlet - Final
                        }
                        else
                        {
                            p[i][j] = -p[i - 1][j]; // pressure outlet - static pressure is zero - Final
                        }
                    }
                } // end for j
            } // end for i
            //................................................................................................

            error = pow(error, 0.5);
            iter = iter + 1;
            if (iter > maxiter)
            {
                break;
            }

        } // end while eps

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
                uold[i][j] = u[i][j];
                u[i][j] = us[i][j] - dt / dx*(p[i + 1][j] - p[i][j]);
            } // end for j
        } // end for i
        //................................................

        // v

        for (int i = 1; i < nx; i++)
        {
            for (int j = 1; j < ny - 1; j++)
            {
                vold[i][j] = v[i][j];
                v[i][j] = vs[i][j] - dt / dy*(p[i][j + 1] - p[i][j]);
            } // end for j
        } // end for i
        int step3_end = clock();
        stepTimingAccumulator["Step 3 - Velocity Update"] += step3_end - step3_start;
        //...............................................................................................

        //...............................................................................................
        // Step 4 - It can be parallelized
        // Solving for temperature
        int step4_start = clock();
        for (int i = 1; i < nx; i++)
        {
            for (int j = 1; j < ny; j++)
            {
                Told[i][j] = T[i][j];
                T[i][j] = T[i][j] + dt*(-0.5*(u[i][j] + u[i - 1][j])*(1.0 / (2.0*dx)*(T[i + 1][j] - T[i - 1][j])) - 0.5*(v[i][j] + v[i][j - 1])*(1.0 / (2.0*dy)*(T[i][j + 1] - T[i][j - 1])) + 1 / (Re*Pr)*(1 / pow(dx, 2.0)*(T[i + 1][j] - 2.0*T[i][j] + T[i - 1][j]) + 1 / pow(dy, 2.0)*(T[i][j + 1] - 2 * T[i][j] + T[i][j - 1])));
            } // end for j
        } // end for i
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
                    T[i][j] = T[i][j + 1]; // bottom wall - Insulated - no flux - Final
                }
                else if (j == ny)
                {
                    T[i][j] = 2.0*(T_0) / T_amb - T[i][j - 1]; // upper boundary - lid with ambient temperature (as air) - Final
                }
                else if (i == 0)
                {
                    if (j*dy < 2.0)
                    {
                        T[i][j] = 2.0*T_L / T_amb - T[i + 1][j]; // left wall at T_L - Constant Temperature - Final
                    }
                    else
                    {
                        T[i][j] = 2.0*T_0 / T_amb - T[i + 1][j]; // left inlet at T_0 (initial temperature) - Final
                    }
                }
                else if (i == nx)
                {
                    if (j*dy < 2.0)
                    {
                        T[i][j] = 2.0*T_R / T_amb - T[i - 1][j]; // right wall at T_R - Final
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

        float TVt, TV, TV2, TV3; TV = 0; TV2 = 0; TV3 = 0; float abs, abs2, abs3;
        for (int i = 1; i < nx - 1; i++)
        {
            for (int j = 1; j < ny - 2; j++)
            {
                abs = v[i][j] - vold[i][j];
                TV = TV + pow(pow(abs, 2), 0.5);
            } // end for i
        } // end for j

        TV = TV / ((nx - 1)*(ny - 2));

        if (TV < st && error < eps)
        {
            cout << "Steady state time = " << t << " (s) " << endl;
            break;
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

    } // end while time

    //........................................................................................................

    // Step 6
    // Co-locate the staggered grid points 
    int step6_start = clock();
    for (int i = 0; i < nx; i++)
    {
        for (int j = 0; j < ny; j++)
        {
            vc[i][j] = 1.0 / 2.0*(v[i + 1][j] + v[i][j]);
            pc[i][j] = 1.0 / 4.0*(p[i][j] + p[i + 1][j] + p[i][j + 1] + p[i + 1][j + 1]);
            uc[i][j] = 1.0 / 2.0*(u[i][j] + u[i][j + 1]);
            om[i][j] = 1.0 / dx*(v[i + 1][j] - v[i][j]) - 1.0 / dy*(u[i][j + 1] - u[i][j]);
            Tc[i][j] = 1.0 / 4.0*(T[i][j] + T[i + 1][j] + T[i][j + 1] + T[i + 1][j + 1]);

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
            f << setw(15) << t - dt << setw(15) << i*dx << setw(15) << j*dy << setw(15) << uc[i][j] << setw(15) << vc[i][j] << setw(15) << pc[i][j] << setw(15) << Tc[i][j] * T_amb - 273.15 << setw(15) << om[i][j] << endl;
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

    return 0;
} // end main
