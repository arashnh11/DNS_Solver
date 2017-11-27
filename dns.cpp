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

#include "freeglut/include/GL/glut.h"

#include "HSV_RGB.h"
#include "DNSCPU.h"
#include "DNSGPU.cuh"

using namespace std;

const int WINDOW_WIDTH = 500;
const int WINDOW_HEIGHT = 500;

struct RenderData2D
{
    uint32_t m_width;
    std::vector<rgb> m_data;

    void Init(std::vector<float>& i_data, uint32_t i_width, float maxvalue, float minValue)
    {
        m_width = i_width;
		//float maxvalue = *std::max_element(i_data.begin(), i_data.end());
        //float minValue = *std::min_element(i_data.begin(), i_data.end());
        m_data.resize(i_data.size());
        for (int i = 0; i < i_data.size(); ++i)
        {
            hsv color;
            color.h = (i_data[i] - minValue) / (maxvalue - minValue) * 360;
            color.s = 0.7;
            color.v = 0.7;
            
            m_data[i] = (hsv2rgb(color));
        }
    }
};

RenderData2D s_drawData;



void idleCPU()
{
	DNSCPU::RunSimulation();
	std::vector<float>& data = DNSCPU::getU();
	float maxvalue = *std::max_element(data.begin(), data.end());
	float minvalue = *std::min_element(data.begin(), data.end());

    maxvalue = 3.0f;
    minvalue = -3.0f;

	s_drawData.Init(data, DNSCPU::getUWidth(), maxvalue, minvalue);
	glutPostRedisplay();
}

void idleGPU()
{
	DNSGPU::RunSimulation();
	std::vector<float>& data = DNSGPU::getU();
	float maxvalue = *std::max_element(data.begin(), data.end());
	float minvalue = *std::min_element(data.begin(), data.end());

	maxvalue = 3.0f;
	minvalue = -3.0f;

	s_drawData.Init(data, DNSGPU::getUWidth(), maxvalue, minvalue);
	glutPostRedisplay();
}

void RenderPrimitive();
void display();

// The original SOR solver lives in DoStuff();
// At the end of DoStuff we pass in the data to be drawn on screen
void DoStuff();

int main(int argc, char**argv)
{
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_SINGLE | GLUT_RGB);
    glutInitWindowSize(500, 500);
    glutInitWindowPosition(100, 100);
    glutCreateWindow("points and lines");
    glutDisplayFunc(display);
    
    
    // Uncomment this to run the original solver. 
    //DoStuff();

    // The idle func (defined in thsi while) will run the simulation (currently its jacobi)
    glutIdleFunc(idleGPU);
	
    glutMainLoop();
    return 0;
} // end main



//void DoStuff()
//{
//    // output format
//    float start_clock = clock();
//    ofstream f("result_cpu.txt"); // Solution Results
//    f.setf(ios::fixed | ios::showpoint);
//    f << setprecision(5);
//
//    ofstream g("convergence_cpu.txt"); // Convergence history
//    g.setf(ios::fixed | ios::showpoint);
//    g << setprecision(5);
//    cout.setf(ios::fixed | ios::showpoint);
//    cout << setprecision(5);
//
//    // Input parameters 
//    float Re, Pr, Fr, T_L, T_0, T_amb, ni, nj, dx, dy, t, ny, nx, eps, beta, iter, maxiter, tf, st, counter, column, u_wind, T_R, Lx, Ly;
//    Lx = 6.0; Ly = 6.0; // Domain dimensions
//    ni = 10.0; // Number of nodes per unit length in x direction
//    nj = 10.0; // Number of nodes per unit length in y direction
//    nx = Lx * ni; ny = Ly * nj; // Number of Nodes in each direction
//    u_wind = 1; // Reference velocity
//    st = 0.00005; // Total variance criteria
//    eps = 0.001; // Pressure convergence criteria
//    tf = 100.0; // Final time step
//    Pr = 0.5*(0.709 + 0.711); // Prandtl number
//    Re = 300.0; Fr = 0.3; // Non-dimensional numbers for inflow conditions
//    dx = Lx / (nx - 1); dy = Ly / (ny - 1); // dx and dy
//    beta = 1.4; // Successive over relaxation factor (SOR)
//    t = 0; // Initial time step
//    T_L = 100.0; // Left wall temperature (C)
//    T_R = 50.0; // Right wall temperature (C)
//    T_amb = 25.0; // Ambient air temperature (C)
//    T_0 = 50.0; // Initial air temperature
//    T_L = T_L + 273.15; T_0 = T_0 + 273.15; T_amb = T_amb + 273.15; T_R = T_R + 273.15;// Unit conversion to (K)
//    maxiter = 100; // Maximum iteration at each time step
//    counter = 0; // initial row for output monitoring
//    column = 1; // Column number for output display
//
//                // Records number of clicks a step takes
//    std::map<string, uint32_t> stepTimingAccumulator;
//
//    // Vectors
//
//    vector<float> u(nx * (ny + 1), 0);
//    vector<float> us(nx*(ny + 1), 0);
//    vector<float> uold(nx * (ny + 1), 0);
//    float wu = ny + 1;
//
//    vector<float> v((nx + 1) * ny, 0);
//    vector<float> vs((nx + 1) * ny, 0);
//    vector<float> vold((nx + 1) * ny, 0);
//    float wv = ny;
//
//    vector<float> p((nx + 1) * (ny + 1), 0);
//    float wp = ny + 1;
//
//    vector<float> T((nx + 1) * (ny + 1), T_0 / T_amb);     // Initializing the flow variable (Temperature)  
//                                                           // Boundary conditions for T (Initialization)
//    float wT = ny + 1;
//
//    vector<float> Told((nx + 1) * (ny + 1), 0);
//    vector<float> om(nx * ny, 0);
//    vector<float> vc(nx * ny, 0);
//    vector<float> uc(nx * ny, 0);
//    vector<float> pc(nx * ny, 0);
//    vector<float> Tc(nx*ny, 0);
//    float wc = ny;
//
//    // Time step size stability criterion
//
//    float mt1 = 0.25*pow(dx, 2.0) / (1.0 / Re); float Rer = 1.0 / Re; float mt2 = 0.25*pow(dy, 2.0) / (1.0 / Re);
//    float dt;
//
//    if (mt1 > Rer)
//    {
//        dt = Rer;
//    }
//    else
//    {
//        dt = mt1;
//    }
//
//    if (dt > mt2)
//    {
//        dt = mt2;
//    }
//
//
//    //......................................................................................
//    // Step 0 - It can be parallelized
//
//    int step0_start = clock();
//    //......................................................................................
//    int step0_end = clock();
//    stepTimingAccumulator["Step 0, Initializing Temperature"] += step0_end - step0_start;
//    //......................................................................................
//
//    // Marching in Time - Outermost loop
//
//    while (t <= tf)
//    {
//
//        iter = 0;
//
//        int stepi1_start = clock();
//        //........................................................................................
//        // Step i1 - it can be parallelized 
//        // boundary conditions for u velocity
//
//        for (int i = 0; i < nx; i++)
//        {
//            for (int j = 0; j < ny + 1; j++)
//            {
//                if (i == 0 && j > 0 && j < ny)
//                {
//                    if (j*dy < 2.0)
//                    {
//                        u[i * wu + j] = 0; // left wall - Final
//                    }
//                    else
//                    {
//                        u[i * wu + j] = u_wind; // left inlet - Final
//                    }
//                }
//                else if (i == nx - 1 && j>0 && j < ny)
//                {
//                    if (j*dy < 2.0)
//                    {
//                        u[i * wu + j] = 0; // Right wall has 0 horizontal velocity - Final
//                    }
//                    else
//                    {
//                        u[i * wu + j] = u[(i - 1) * wu + j]; // right outlet - no velocity change
//                    }
//                }
//                else if (j == 0)
//                {
//                    u[i * wu + j] = -u[i * wu + j + 1]; // bottom ghost - Final
//                }
//                else if (j == ny)
//                {
//                    u[i * wu + j] = u[i * wu + j - 1]; // upper ghost - Final
//                }
//            } // end for j
//        } // end for i
//        int stepi1_end = clock();
//        stepTimingAccumulator["Step i1 - Set Horizontal Velocity Boundary Conditions"] += stepi1_end - stepi1_start;
//        //...............................................................................................
//
//
//        //.........................................................................................
//        // Step i2 - it can be parallelized
//        // boundary conditions for v velocity
//        int stepi2_start = clock();
//
//        for (int i = 0; i < nx + 1; i++)
//        {
//            for (int j = 0; j < ny; j++)
//            {
//                if (j == 0 && i > 0 && i < nx)
//                {
//                    v[i * wv + j] = 0; // bottom wall - Final
//                }
//                else if (j == ny - 1 && i > 0 && i < nx)
//                {
//                    v[i * wv + j] = v[i * wv + j - 1]; // upper wall - Final
//                }
//                else if (i == 0)
//                {
//                    v[i * wv + j] = -v[(i + 1) * wv + j]; // left ghost (Left Wall and inlet has 0 vertical velocity) - Final
//                }
//                else if (i == nx)
//                {
//                    if (j*dy < 2.0)
//                    {
//                        v[i * wv + j] = -v[(i - 1) * wv + j]; // right ghost (Right wall has 0 vertical velocity) - Final
//                    }
//                    else
//                    {
//                        v[i * wv + j] = v[(i - 1) * wv + j]; // right outlet - no velocity gradient
//                    }
//                }
//            } // end for j
//        } // end for I
//        int stepi2_end = clock();
//        stepTimingAccumulator["Step i2 - Set Vertical Velocity Boundary Conditions"] += stepi2_end - stepi2_start;
//        //...............................................................................................
//
//        //...............................................................................................
//        int step1_start = clock();
//        //.........................................................................................
//        // Step 1 - it can be parallelized - Solve for intermediate velocity values
//
//        // u - us - vh - a 
//
//        for (int i = 1; i < nx - 1; i++)
//        {
//            for (int j = 1; j < ny; j++)
//            {
//                float vh = 1.0 / 4.0*(v[i * wv + j] + v[(i + 1) * wv + j] + v[i * wv + j - 1] + v[(i + 1) * wv + j - 1]); // v hat
//                float a = u[i * wu + j] * 1.0 / (2.0*dx)*(u[(i + 1) * wu + j] - u[(i - 1) * wu + j]) + vh*1.0 / (2.0*dy)*(u[i * wu + j + 1] - u[i * wu + j - 1]); // a
//                us[i * wu + j] = dt / Re*(1.0 / pow(dx, 2.0)*(u[(i + 1) * wu + j] - 2.0*u[i * wu + j] + u[(i - 1) * wu + j]) + 1.0 / pow(dy, 2.0)*(u[i * wu + j + 1] - 2.0*u[i * wu + j] + u[i * wu + j - 1])) - a*dt + u[i * wu + j]; // u star
//            } // end for j
//        } // end for i
//
//          //..........................................................................................
//          // Step 1 - it can be parallelized
//          // v - vs - uh - b
//        for (int i = 1; i < nx; i++)
//        {
//            for (int j = 1; j < ny - 1; j++)
//            {
//                float uh = 1.0 / 4.0*(u[i * wu + j] + u[i * wu + j + 1] + u[(i - 1) * wu + j] + u[(i - 1) * wu + j + 1]);
//                float b = uh*1.0 / (2.0*dx)*(v[(i + 1) * wv + j] - v[(i - 1) * wv + j]) + v[i * wv + j] * 1.0 / (2.0*dy)*(v[i * wv + j + 1] - v[i * wv + j - 1]); // b
//                vs[i * wv + j] = dt / Re*(1.0 / pow(dx, 2.0)*(v[(i + 1) * wv + j] - 2.0*v[i * wv + j] + v[(i - 1) * wv + j]) + 1.0 / pow(dy, 2.0)*(v[i * wv + j + 1] - 2.0*v[i * wv + j] + v[i * wv + j - 1])) + dt / pow(Fr, 2.0)*(0.5*(T[i * wT + j] + T[i * wT + j - 1]) - 1) / (0.5*(T[i * wT + j] + T[i * wT + j - 1])) - b*dt + v[i * wv + j]; // v 
//            } // end for j
//        } // end for i
//
//          //...........................................................................................
//          // vs and us on Boundary conditions
//
//        for (int i = 0; i < nx; i++)
//        {
//            us[i * wu + 0] = -us[i * wu + 1]; // bottom ghost - Final
//        } // end for j
//
//          //...........................................................................................
//        for (int j = 0; j < ny + 1; j++)
//        {
//            if (j*dy < 2.0)
//            {
//                us[0 * wu + j] = 0; // left wall - FInal
//                us[(nx - 1) * wu + j] = 0; // right wall - Final
//            }
//            else
//            {
//                us[0 * wu + j] = u_wind; // left inlet - Final
//            }
//        }
//        //...........................................................................................
//
//        for (int j = 0; j < ny; j++)
//        {
//            vs[0 * wv + j] = -vs[1 * wv + j]; // left ghost (Both wall and inlet have 0 vs) - Final
//            if (j*dy < 2.0)
//            {
//                vs[nx * wv + j] = -vs[(nx - 1) * wv + j]; // right ghost (Only the right wall - Final
//            }
//            else
//            {
//                vs[nx * wv + j] = vs[(nx - 1) * wv + j]; // right outlet - no flux
//            }
//        }
//        //............................................................................................
//
//        for (int i = 0; i < nx + 1; i++)
//        {
//            vs[i * wv + 0] = 0; // Bottom wall - Final
//        } // end for i
//          //............................................................................................
//
//        int step1_end = clock();
//        stepTimingAccumulator["Step 1 - Solve for intermediate velocities"] += step1_end - step1_start;
//
//        //...............................................................................................
//        // Step 2 - It can be parallelized 
//        // This is the most expensive part of the code
//        // Poisson equation for pressure
//        int step2_start = clock();
//
//        float error = 1; iter = 0;
//        float diffp, pold;
//        // Solve for pressure iteratively until it converges - Using Gauss Seidel SOR 
//        while (error > eps)
//        {
//            error = 0;
//
//            //............................................................................................
//            for (int i = 1; i < nx; i++)
//            {
//                for (int j = 1; j < ny; j++)
//                {
//                    pold = p[i * wp + j];
//                    p[i * wp + j] = beta*pow(dx, 2.0)*pow(dy, 2.0) / (-2.0*(pow(dx, 2.0) + pow(dy, 2.0)))*(-1.0 / pow(dx, 2.0)*(p[(i + 1) * wp + j] + p[(i - 1) * wp + j] + p[i * wp + j + 1] + p[i * wp + j - 1]) + 1.0 / dt*(1.0 / dx*(us[i * wu + j] - us[(i - 1) * wu + j]) + 1.0 / dy*(vs[i * wv + j] - vs[i * wv + j - 1]))) + (1.0 - beta)*p[i * wp + j];
//                    diffp = pow((p[i * wp + j] - pold), 2.0);
//                    error = error + diffp;
//                } // end for j
//            } // end for i
//              //............................................................................................
//              // boundary conditions for pressure
//
//            for (int i = 0; i < nx + 1; i++)
//            {
//                for (int j = 0; j < ny + 1; j++)
//                {
//                    if (j == 0)
//                    {
//                        p[i * wp + j] = p[i * wp + j + 1]; // bottom wall - Final
//                    }
//                    else if (j == ny)
//                    {
//                        p[i * wp + j] = p[i * wp + j - 1]; // Upper - no flux
//                    }
//                    else if (i == 0)
//                    {
//                        if (j*dy < 2.0)
//                        {
//                            p[i * wp + j] = p[(i + 1) * wp + j]; // left wall - not the inlet - Final
//                        }
//                        else
//                        {
//                            p[i * wp + j] = p[(i + 1) * wp + j];
//                        }
//                    }
//                    else if (i == nx)
//                    {
//                        if (j*dy < 2.0)
//                        {
//                            p[i * wp + j] = p[(i - 1) * wp + j]; // right wall - not the outlet - Final
//                        }
//                        else
//                        {
//                            p[i * wp + j] = -p[(i - 1) * wp + j]; // pressure outlet - static pressure is zero - Final
//                        }
//                    }
//                } // end for j
//            } // end for i
//              //................................................................................................
//
//            error = pow(error, 0.5);
//            iter = iter + 1;
//            if (iter > maxiter)
//            {
//                break;
//            }
//
//        } // end while eps
//
//        int step2_end = clock();
//        stepTimingAccumulator["Step 2 - Solve for pressure until tolerance or max iterations"] += step2_end - step2_start;
//        //...............................................................................................
//
//        //.................................................................................................
//        // Step 3 - It can be parallelized 
//        // velocity update - projection method
//        int step3_start = clock();
//
//        // u
//
//        for (int i = 1; i < nx - 1; i++)
//        {
//            for (int j = 1; j < ny; j++)
//            {
//                uold[i * wu + j] = u[i * wu + j];
//                u[i * wu + j] = us[i * wu + j] - dt / dx*(p[(i + 1) * wp + j] - p[i * wp + j]);
//            } // end for j
//        } // end for i
//          //................................................
//
//          // v
//
//        for (int i = 1; i < nx; i++)
//        {
//            for (int j = 1; j < ny - 1; j++)
//            {
//                vold[i * wv + j] = v[i * wv + j];
//                v[i * wv + j] = vs[i * wv + j] - dt / dy*(p[i * wp + j + 1] - p[i * wp + j]);
//            } // end for j
//        } // end for i
//        int step3_end = clock();
//        stepTimingAccumulator["Step 3 - Velocity Update"] += step3_end - step3_start;
//        //...............................................................................................
//
//        //...............................................................................................
//        // Step 4 - It can be parallelized
//        // Solving for temperature
//        int step4_start = clock();
//        Told = T;
//        for (int i = 1; i < nx; i++)
//        {
//            for (int j = 1; j < ny; j++)
//            {
//
//                T[i * wT + j] = Told[i * wT + j] + dt*(-0.5*(u[i * wu + j] + u[(i - 1) * wu + j])*(1.0 / (2.0*dx)*(Told[(i + 1) * wT + j] - Told[(i - 1) * wT + j])) - 0.5*(v[i * wv + j] + v[i * wv + j - 1])*(1.0 / (2.0*dy)*(Told[i * wT + j + 1] - Told[i * wT + j - 1])) + 1 / (Re*Pr)*(1 / pow(dx, 2.0)*(Told[(i + 1) * wT + j] - 2.0*Told[i * wT + j] + Told[(i - 1) * wT + j]) + 1 / pow(dy, 2.0)*(Told[i * wT + j + 1] - 2 * Told[i * wT + j] + Told[i * wT + j - 1])));
//            } // end for j
//        } // end for i
//
//        int step4_end = clock();
//        stepTimingAccumulator["Step 4 - Solving for temperature"] += step4_end - step4_start;
//        //................................................................................................
//
//        //...............................................................................................
//        // Step i3 - Initializing boundary conditions for temperature 
//        // boundary conditions for Temperature
//        int stepi3_start = clock();
//
//        for (int i = 0; i < nx + 1; i++)
//        {
//            for (int j = 0; j < ny + 1; j++)
//            {
//                if (j == 0)
//                {
//                    T[i * wT + j] = T[i * wT + j + 1]; // bottom wall - Insulated - no flux - Final
//                }
//                else if (j == ny)
//                {
//                    T[i * wT + j] = 2.0*(T_0) / T_amb - T[i * wT + j - 1]; // upper boundary - lid with ambient temperature (as air) - Final
//                }
//                else if (i == 0)
//                {
//                    if (j*dy < 2.0)
//                    {
//                        T[i * wT + j] = 2.0*T_L / T_amb - T[(i + 1) * wT + j]; // left wall at T_L - Constant Temperature - Final
//                    }
//                    else
//                    {
//                        T[i * wT + j] = 2.0*T_0 / T_amb - T[(i + 1) * wT + j]; // left inlet at T_0 (initial temperature) - Final
//                    }
//                }
//                else if (i == nx)
//                {
//                    if (j*dy < 2.0)
//                    {
//                        T[i * wT + j] = 2.0*T_R / T_amb - T[(i - 1) * wT + j]; // right wall at T_R - Final
//                    }
//                }
//            } // end for j
//        } // end for i
//        int stepi3_end = clock();
//        stepTimingAccumulator["Step i3 - Initializing boundary conditions for temperature"] += stepi3_end - stepi3_start;
//        //...............................................................................................
//
//        //...............................................................................................
//        // Step 5 - Checking if solution reached steady state
//        // Checking the steady state condition
//        int step5_start = clock();
//
//        float TV, diffv; TV = 0;
//        for (int i = 1; i < nx - 1; i++)
//        {
//            for (int j = 1; j < ny - 2; j++)
//            {
//                diffv = v[i * wv + j] - vold[i * wv + j];
//                TV = TV + pow(pow(diffv, 2), 0.5);
//            } // end for i
//        } // end for j
//
//        TV = TV / ((nx - 1)*(ny - 2));
//
//        if (TV < st && error < eps)
//        {
//            cout << "Steady state time = " << t << " (s) " << endl;
//            break;
//        }
//        counter = counter + 1;
//        if (fmod(counter, 10) == 0 || counter == 1)
//        {
//            //cout << "" << endl;
//            //cout << "Column" << setw(30) << "time(s)" << setw(30) << "Iterations on Pressure" << setw(30) << "Pressure Residual" << setw(30) << "Total Variance" << endl;
//        } // end if
//        int step5_end = clock();
//        stepTimingAccumulator["Step 5 - Check for steady state"] += step5_end - step5_start;
//        //...............................................................................................
//
//
//        //cout << column << setw(30) << t << setw(30) << iter << setw(30) << error << setw(30) << TV << endl;
//        g << column << setw(30) << t << setw(30) << iter << setw(30) << error << setw(30) << TV << endl;
//        t = t + dt;
//        column = column + 1;
//
//    } // end while time
//
//      //........................................................................................................
//
//      // Step 6
//      // Co-locate the staggered grid points 
//    int step6_start = clock();
//    for (int i = 0; i < nx; i++)
//    {
//        for (int j = 0; j < ny; j++)
//        {
//            vc[i * wc + j] = 1.0 / 2.0*(v[(i + 1) * wv + j] + v[i * wv + j]);
//            pc[i * wc + j] = 1.0 / 4.0*(p[i * wp + j] + p[(i + 1) * wp + j] + p[i * wp + j + 1] + p[(i + 1) * wp + j + 1]);
//            uc[i * wc + j] = 1.0 / 2.0*(u[i*wu + j] + u[i * wu + j + 1]);
//            om[i * wc + j] = 1.0 / dx*(v[(i + 1) * wv + j] - v[i * wv + j]) - 1.0 / dy*(u[i * wu + j + 1] - u[i * wu + j]);
//            Tc[i * wc + j] = 1.0 / 4.0*(T[i * wT + j] + T[(i + 1) * wT + j] + T[i * wT + j + 1] + T[(i + 1) * wT + j + 1]);
//        } // end for j
//    } // end for i
//      //........................................................................................................
//    int step6_end = clock();
//    stepTimingAccumulator["Step 6 - Co-locate staggered grid points"] += step6_end - step6_start;
//
//    // Steady state results
//
//    for (int j = 0; j < ny; j++)
//    {
//        for (int i = 0; i < nx; i++)
//        {
//            f << setw(15) << t - dt << setw(15) << i*dx << setw(15) << j*dy << setw(15) << uc[i * wc + j] << setw(15) << vc[i * wc + j] << setw(15) << pc[i * wc + j] << setw(15) << Tc[i * ny + j] * T_amb - 273.15 << setw(15) << om[i * wc + j] << endl;
//        } // end for i
//    } // end for j
//      //.........................................................................................................
//
//    float end_clock = clock();
//    cout << "CPU time = " << (end_clock - start_clock) / CLOCKS_PER_SEC << " (s)" << endl;
//    //cout << "Re = " << Re << endl;
//    //cout << "Fr = " << Fr << endl;
//
//    for (auto it = stepTimingAccumulator.begin(); it != stepTimingAccumulator.end(); it++)
//    {
//        float seconds = (float)it->second / CLOCKS_PER_SEC;
//        std::cout << it->first << "\t" << seconds << endl;
//    }
//
//    std::vector<float> toDraw(u);
//    s_drawData.Init(toDraw, wu, *(std::max_element(toDraw.begin(), toDraw.end())), *(std::min_element(toDraw.begin(), toDraw.end())));
//
//}

void renderPrimitive()
{
    //     glColor3f(1, 0.4, 0.2);
    //glPointSize(WINDOW_WIDTH/s_drawData.m_width);
    int windowWidth = glutGet(GLUT_WINDOW_WIDTH);
    int windowHeight = glutGet(GLUT_WINDOW_HEIGHT);
    if (s_drawData.m_data.size() > 0)
    {
        glPointSize(windowWidth / s_drawData.m_width);
        glBegin(GL_POINTS);
        int height = s_drawData.m_data.size() / s_drawData.m_width;
        for (int i = 0; i < s_drawData.m_data.size(); ++i)
        {
            float x = i / s_drawData.m_width;
            float y = i - x*s_drawData.m_width;
            rgb color = s_drawData.m_data[i];
            glColor3f(color.r, color.g, color.b);

            x = (2 * x / s_drawData.m_width) - 1;
            y = (2 * y / height) - 1;
            glVertex2d(x, y);
        }
        glEnd();
    }

}

void display()
{
    glClearColor(0.3f, 0.3f, 0.3f, 0.3f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(-2, 2, -2, 2, -1, 1);

    glPushMatrix();
    glTranslatef(0.0f, 0.0f, -0.5f);
    renderPrimitive();
    glPopMatrix();

    glutSwapBuffers();
}