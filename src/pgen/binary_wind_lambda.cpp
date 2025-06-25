//======================================================================================
// Athena++ astrophysical MHD code
// Copyright (C) 2014 James M. Stone  <jmstone@princeton.edu>
//
// This program is free software: you can redistribute and/or modify it under the terms
// of the GNU General Public License (GPL) as published by the Free Software Foundation,
// either version 3 of the License, or (at your option) any later version.
//
// This program is distributed in the hope that it will be useful, but WITHOUT ANY
// WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A 
// PARTICULAR PURPOSE.  See the GNU General Public License for more details.
//
// You should have received a copy of GNU GPL in the file LICENSE included in the code
// distribution.  If not see <http://www.gnu.org/licenses/>.
//======================================================================================
//! \file planet_wind_lambda.cpp: tidal perturbation of planet wind defined by hydrodynamic escape parameter lambda
//======================================================================================

// C++ headers
#include <sstream>
#include <cmath>
#include <stdexcept>
#include <fstream>
#include <iostream>

// Athena++ headers
#include "../athena.hpp"
#include "../globals.hpp"
#include "../athena_arrays.hpp"
#include "../parameter_input.hpp"
#include "../coordinates/coordinates.hpp"
#include "../eos/eos.hpp"
#include "../field/field.hpp"
#include "../hydro/hydro.hpp"
#include "../mesh/mesh.hpp"
#include "../bvals/bvals.hpp"
#include "../utils/utils.hpp"
#include "../outputs/outputs.hpp"


void BinaryWind(MeshBlock *pmb, const Real time, const Real dt, const AthenaArray<Real> *flux,
                  const AthenaArray<Real> &prim, const AthenaArray<Real> &bcc, AthenaArray<Real> &cons);


void cross(Real (&A)[3],Real (&B)[3],Real (&AxB)[3]);

Real fspline(Real r, Real eps);
Real pspline(Real r, Real eps);

Real PhiEff(Real x, Real y, Real z);
Real PhiL1();


// global (to this file) problem parameters
Real gamma_gas; 

Real Ggrav;   // G 
Real GM2, GM1; // point masses
Real rsoft; // softening length of PM 2

Real x1i[3], v1i[3], x2i[3], v2i[3]; // cartesian positions/vels of the secondary object, gas->particle acceleration
Real Omega[3];  // vector rotation of the frame, initial wind
Real sma;

Real rho_surface, lambda; // planet surface variables
Real phi_critical;

//======================================================================================
//! \fn void Mesh::InitUserMeshData(ParameterInput *pin)
//  \brief Function to initialize problem-specific data in mesh class.  Can also be used
//  to initialize variables which are global to (and therefore can be passed to) other
//  functions in this file.  Called in Mesh constructor.
//======================================================================================

void Mesh::InitUserMeshData(ParameterInput *pin)
{

  // read in some global params (to this file)

  gamma_gas = pin->GetReal("hydro","gamma");

  Ggrav = pin->GetOrAddReal("problem","Ggrav",6.67408e-8);
  GM2 = pin->GetOrAddReal("problem","M2",0.5)*Ggrav;
  GM1 = pin->GetOrAddReal("problem","M1",0.5)*Ggrav;

  rsoft = pin->GetOrAddReal("problem","rsoft",0.1);
  
  rho_surface = pin->GetOrAddReal("problem","rho_surface",1.0);
  lambda = pin->GetOrAddReal("problem","lambda",5.0);

  sma = pin->GetOrAddReal("problem","sma",1.0);
  Real phi_crit_o_phi_L1 = pin->GetOrAddReal("problem","phi_critical_o_phi_L1",1.0);
  Real Omega_orb, vcirc;
 

  // Enroll a Source Function
  EnrollUserExplicitSourceFunction(BinaryWind);


  // PARTICLES
  //Real vcirc = sqrt((GM1+GM2)/sma + accel*sma);    
  vcirc = sqrt((GM1+GM2)/sma);
  Omega_orb = vcirc/sma;

  // set the initial conditions for the pos/vel of the binary
  x1i[0] = -sma*(GM2/(GM1+GM2));
  x1i[1] = 0.0;
  x1i[2] = 0.0;
  
  v1i[0] = 0.0;
  v1i[1]=  0.0; //- vcirc*(GM2/(GM1+GM2)); 
  v1i[2] = 0.0;
  
  x2i[0] = sma*(GM2/(GM1+GM2));
  x2i[1] = 0.0;
  x2i[2] = 0.0;
  
  v2i[0] = 0.0;
  v2i[1]=  0.0; //vcirc*(GM2/(GM1+GM2)); 
  v2i[2] = 0.0;
  
  // now set the initial condition for Omega
  Omega[0] = 0.0;
  Omega[1] = 0.0;
  Omega[2] = Omega_orb;

  
  Real phi_L1 = PhiL1();
  phi_critical = phi_crit_o_phi_L1*phi_L1;
  
  // Print out some info
  if (Globals::my_rank==0){
    std::cout << "==========================================================\n";
    std::cout << "==========   SIMULATION INFO =============================\n";
    std::cout << "==========================================================\n";
    std::cout << "time =" << time << "\n";
    std::cout << "Ggrav = "<< Ggrav <<"\n";
    std::cout << "gamma = "<< gamma_gas <<"\n";
    std::cout << "GM1 = "<< GM1 <<"\n";
    std::cout << "GM2 = "<< GM2 <<"\n";
    std::cout << "Omega_orb="<< Omega_orb << "\n";
    std::cout << "a = "<< sma <<"\n";
    std::cout << "P = "<< 6.2832*sqrt(sma*sma*sma/(GM1+GM2)) << "\n";
    std::cout << "rsoft ="<<rsoft<<"\n";
    std::cout << "==========================================================\n";
    std::cout << "==========   BC INFO         =============================\n";
    std::cout << "==========================================================\n";
    std::cout << "rho_surface = "<< rho_surface <<"\n";
    std::cout << "press_surface = "<< -rho_surface*phi_critical/(gamma_gas*lambda) <<"\n";
    std::cout << "lambda = "<< lambda <<"\n";
    std::cout << "phi_critical ="<<phi_critical<<"\n";
    std::cout << "phi_critical/phi_L1 ="<<phi_critical/phi_L1<<"\n";
    std::cout << "==========================================================\n";
    std::cout << "==========   Particles       =============================\n";
    std::cout << "==========================================================\n";
    std::cout << "x1 ="<<x1i[0]<<"\n";
    std::cout << "y1 ="<<x1i[1]<<"\n";
    std::cout << "z1 ="<<x1i[2]<<"\n";
    std::cout << "vx1 ="<<v1i[0]<<"\n";
    std::cout << "vy1 ="<<v1i[1]<<"\n";
    std::cout << "vz1 ="<<v1i[2]<<"\n";
    std::cout << "x2 ="<<x2i[0]<<"\n";
    std::cout << "y2 ="<<x2i[1]<<"\n";
    std::cout << "z2 ="<<x2i[2]<<"\n";
    std::cout << "vx2 ="<<v2i[0]<<"\n";
    std::cout << "vy2 ="<<v2i[1]<<"\n";
    std::cout << "vz2 ="<<v2i[2]<<"\n";
    std::cout << "==========================================================\n";
  }
  
} // end








//======================================================================================
//! \fn void MeshBlock::ProblemGenerator(ParameterInput *pin)
//  \brief Spherical Coords HSE Envelope problem generator
//======================================================================================

void MeshBlock::ProblemGenerator(ParameterInput *pin)
{

  // local vars
  Real den, pres, vr;

  // SETUP THE INITIAL CONDITIONS ON MESH
  for (int k=ks; k<=ke; k++) {
    for (int j=js; j<=je; j++) {
      for (int i=is; i<=ie; i++) {

	Real x = pcoord->x1v(i);
	Real y = pcoord->x2v(j);
	Real z = pcoord->x3v(k);

	Real phi = PhiEff(x,y,z);

	// location relative to point masses
	Real d1  = sqrt(pow(x-x1i[0], 2) +
			pow(y-x1i[1], 2) +
			pow(z-x1i[2], 2) );
	Real d2  = sqrt(pow(x-x2i[0], 2) +
			pow(y-x2i[1], 2) +
			pow(z-x2i[2], 2) );


	Real press_surface = -rho_surface*phi_critical/(gamma_gas*lambda);
	Real cs = std::sqrt(gamma_gas*press_surface/rho_surface);
	Real vx,vy,vz;

	Real Rroche = 0.379*sma; // for q=1
	

	if(phi < phi_critical and d1 <= sma/2.){
	  den = rho_surface;
	  pres = press_surface;
	  vx = 0.0;
	  vy = 0.0;
	  vz = 0.0;
	}else if(phi< phi_critical and d2 <= sma/2.){
	  den = rho_surface;
	  pres = press_surface;
	  vx = 0.0;
	  vy = 0.0;
	  vz = 0.0;
	}else{
	  den = rho_surface * ( pow((d1/Rroche),-2) +  pow((d2/Rroche),-2) );
	  pres = press_surface * pow(den / rho_surface, gamma_gas);
	  vx = ((x-x1i[0])/d1 + (x-x2i[0])/d2)*cs/2.0;  // wind directed outward from each at v=cs
	  vy = ((y-x1i[1])/d1 + (y-x2i[1])/d2)*cs/2.0;
	  vz = ((z-x1i[2])/d1 + (z-x2i[2])/d2)*cs/2.0;
	}

	
	phydro->u(IDN,k,j,i) = den;
	phydro->u(IM1,k,j,i) = den*vx;
	phydro->u(IM2,k,j,i) = den*vy;
	phydro->u(IM3,k,j,i) = den*vz;
	phydro->u(IEN,k,j,i) = pres/(gamma_gas-1.0);
	phydro->u(IEN,k,j,i) += 0.5*(SQR(phydro->u(IM1,k,j,i))+SQR(phydro->u(IM2,k,j,i))
				     + SQR(phydro->u(IM3,k,j,i)))/phydro->u(IDN,k,j,i);

      }
    }
  } // end loop over cells
  return;
} // end ProblemGenerator






// Source Function for two point masses
void BinaryWind(MeshBlock *pmb, const Real time, const Real dt, const AthenaArray<Real> *flux,
		  const AthenaArray<Real> &prim, const AthenaArray<Real> &bcc, AthenaArray<Real> &cons)
{   
  // Gravitational acceleration from orbital motion
  for (int k=pmb->ks; k<=pmb->ke; k++) {
    Real z= pmb->pcoord->x3v(k);
    for (int j=pmb->js; j<=pmb->je; j++) {
      Real y= pmb->pcoord->x2v(j);
      for (int i=pmb->is; i<=pmb->ie; i++) {
	Real x = pmb->pcoord->x1v(i);
		// distances to zone
	Real d1  = sqrt(pow(x-x1i[0], 2) +
			pow(y-x1i[1], 2) +
			pow(z-x1i[2], 2) );
	Real d2  = sqrt(pow(x-x2i[0], 2) +
			pow(y-x2i[1], 2) +
			pow(z-x2i[2], 2) );
  
	//
	//  COMPUTE ACCELERATIONS 
	//
	// PM1,2 gravitational accels in cartesian coordinates
	Real a_x = - GM1*fspline(d1,rsoft)*(x-x1i[0]) - GM2*fspline(d2,rsoft)*(x-x2i[0]);   
	Real a_y = - GM1*fspline(d1,rsoft)*(y-x1i[1]) - GM2*fspline(d2,rsoft)*(y-x2i[1]);  
	Real a_z = - GM1*fspline(d1,rsoft)*(z-x1i[2]) - GM2*fspline(d2,rsoft)*(z-x2i[2]);


	// Coriolis & Centrifugal
	// distance from the origin in cartesian (vector)
	Real rxyz[3];
	rxyz[0] = x;
	rxyz[1] = y;
	rxyz[2] = z;

	// gas velocity (vector)
	Real vgas[3];
	vgas[0] = prim(IVX,k,j,i);
	vgas[1] = prim(IVY,k,j,i);
	vgas[2] = prim(IVZ,k,j,i);

	// centrifugal
	Real Omega_x_r[3], Omega_x_Omega_x_r[3];
	cross(Omega,rxyz,Omega_x_r);
	cross(Omega,Omega_x_r,Omega_x_Omega_x_r);
	
	a_x += - Omega_x_Omega_x_r[0];
	a_y += - Omega_x_Omega_x_r[1];
	a_z += - Omega_x_Omega_x_r[2];
	
	  // coriolis
	Real Omega_x_v[3];
	cross(Omega,vgas,Omega_x_v);
	
	a_x += -2.0*Omega_x_v[0];
	a_y += -2.0*Omega_x_v[1];
	a_z += -2.0*Omega_x_v[2];
	
	//
	// ADD SOURCE TERMS TO THE GAS MOMENTA/ENERGY
	//
	Real den = prim(IDN,k,j,i);
	
	Real src_1 = dt*den*a_x; 
	Real src_2 = dt*den*a_y;
	Real src_3 = dt*den*a_z;
	
	// add the source term to the momenta  (source = - rho * a)
	cons(IM1,k,j,i) += src_1;
	cons(IM2,k,j,i) += src_2;
	cons(IM3,k,j,i) += src_3;
	
	// update the energy (source = - rho v dot a)
	cons(IEN,k,j,i) +=  src_1*prim(IVX,k,j,i) + src_2*prim(IVY,k,j,i) + src_3*prim(IVZ,k,j,i);


	
	// STAR BOUNDARIES (note, overwrites the grav accel, ie gravitational accel is not applied in this region)
	Real phi = PhiEff(x,y,z);
	if(phi < phi_critical and ( d1 <= sma/2. or  d2 <= sma/2.)  ){
	  Real press_surface = -rho_surface*phi_critical/(gamma_gas*lambda);
	  cons(IDN,k,j,i) = rho_surface;
	  cons(IM1,k,j,i) = 0.0;
	  cons(IM2,k,j,i) = 0.0;
	  cons(IM3,k,j,i) = 0.0;
	  cons(IEN,k,j,i) = press_surface/(gamma_gas-1.0);
	  
	}


      }
    }
  } // end loop over cells



}



  

Real fspline(Real r, Real eps){
  // Hernquist & Katz 1989 spline kernel F=-GM r f(r,e) EQ A2
  Real u = r/eps;
  Real u2 = u*u;

  if (u<1.0){
    return pow(eps,-3) * (4./3. - 1.2*pow(u,2) + 0.5*pow(u,3) );
  } else if(u<2.0){
    return pow(r,-3) * (-1./15. + 8./3.*pow(u,3) - 3.*pow(u,4) + 1.2*pow(u,5) - 1./6.*pow(u,6));
  } else{
    return pow(r,-3);
  }

}


Real pspline(Real r, Real eps){
  Real u = r/eps;
  if (u<1.0){
    return -2/eps *(1./3.*pow(u,2) -0.15*pow(u,4) + 0.05*pow(u,5)) +7./(5.*eps);
  } else if(u<2.0){
    return -1./(15.*r) - 1/eps*( 4./3.*pow(u,2) - pow(u,3) + 0.3*pow(u,4) -1./30.*pow(u,5)) + 8./(5.*eps);
  } else{
    return 1/r;
  }

}


void cross(Real (&A)[3],Real (&B)[3],Real (&AxB)[3]){
  // set the vector AxB = A x B
  AxB[0] = A[1]*B[2] - A[2]*B[1];
  AxB[1] = A[2]*B[0] - A[0]*B[2];
  AxB[2] = A[0]*B[1] - A[1]*B[0];
}



Real PhiEff(Real x, Real y, Real z){
  Real d1  = sqrt(pow(x-x1i[0], 2) +
		  pow(y-x1i[1], 2) +
		  pow(z-x1i[2], 2) );
  Real d2  = sqrt(pow(x-x2i[0], 2) +
		  pow(y-x2i[1], 2) +
		  pow(z-x2i[2], 2) );
  Real Rcyl = sqrt(x*x + y*y);
  
  return -GM1*pspline(d1,rsoft) - GM2*pspline(d2,rsoft) - 0.5*Omega[2]*Omega[2]*Rcyl*Rcyl;
}

Real PhiL1(){
  int n = 1000;
  Real x,x_L1;
  Real phi;
  Real phi_max=-1.e99;

  for(int i=0;i<n;i++){
    x = x1i[0] + (x2i[0]-x1i[0])*i/n;
    phi = PhiEff(x,0,0);
    //std::cout << x<< phi << "\n";
    phi_max = std::max(phi,phi_max);
    if(phi==phi_max){
      x_L1 = x;
    }
  }
  
  if (Globals::my_rank==0){
    std::cout << "==========================================================\n";
    std::cout << "xL1 ="<<x_L1<<" phiL1 = "<<phi_max <<"\n";
    std::cout << "==========================================================\n";
  }
  return phi_max;
}
