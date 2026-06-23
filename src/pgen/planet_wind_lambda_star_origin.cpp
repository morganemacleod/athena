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
#include "../scalars/scalars.hpp"

void DiodeOuterX1(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,FaceField &b,
		 Real time, Real dt, int is, int ie, int js, int je, int ks, int ke, int ngh);

void WindInnerX1(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,FaceField &b,
		 Real time, Real dt, int is, int ie, int js, int je, int ks, int ke, int ngh);

void AccreteInnerX1(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,FaceField &b,
		 Real time, Real dt, int is, int ie, int js, int je, int ks, int ke, int ngh);



void StarPlanetWinds(MeshBlock *pmb, const Real time, const Real dt,  const AthenaArray<Real> *flux,
		  const AthenaArray<Real> &prim,
		  const AthenaArray<Real> &prim_scalar, const AthenaArray<Real> &bcc,
		  AthenaArray<Real> &cons, AthenaArray<Real> &cons_scalar); 


void ParticleAccels(Real (&xi)[3],Real (&vi)[3],Real (&ai)[3]);
void particle_step(Real dt,Real (&xi)[3],Real (&vi)[3],Real (&ai)[3]);
void kick(Real dt,Real (&xi)[3],Real (&vi)[3],Real (&ai)[3]);
void drift(Real dt,Real (&xi)[3],Real (&vi)[3],Real (&ai)[3]);

void cross(Real (&A)[3],Real (&B)[3],Real (&AxB)[3]);

void WritePMTrackfile(Mesh *pm, ParameterInput *pin);

Real MdotMeshStar(MeshBlock *pmb, int iout);
Real MdotMeshPlanet(MeshBlock *pmb, int iout);

Real fspline(Real r, Real eps);
Real pspline(Real r, Real eps);
Real press_factor_aniso( Real d2, Real solar_angle);


//int RefinementCondition(MeshBlock *pmb);


// global (to this file) problem parameters
Real gamma_gas; 
//Real da,pa; // ambient density, pressure

Real Ggrav;   // G 
Real GM2, GM1; // point masses
Real rsoft2; // softening length of PM 2
int corotating_frame; // flags for output, gas backreaction on EOM, frame choice
int n_particle_substeps; // substepping of particle integration

Real xi[3], vi[3]; // cartesian positions/vels of the secondary object, gas->particle acceleration
Real Omega[3];  // vector rotation of the frame, initial wind

Real trackfile_next_time, trackfile_dt;
int  trackfile_number;

int is_restart;

Real rho_surface_star, lambda_star; // star surface variables
Real r_inner;

Real rho_surface_planet, lambda_planet, radius_planet; //planet surface variables
bool aniso_heat;
Real rp_aniso_inner;
Real p_night;
Real omega_planet, omega_star; // rotation of planet and star boundaries

//bool initialize_planet_wind; // true=planetary wind backgorund ic, false stellar wind ic background
Real da,pa;

//Real x1_min_derefine; // for AMR
//Real threshold, dscale; // for AMR

int star_mode; // setting for the stellar BC
Real vr_planet, vr_star;

Real scalar_val=1.e-10;



//======================================================================================
//! \fn void Mesh::InitUserMeshData(ParameterInput *pin)
//  \brief Function to initialize problem-specific data in mesh class.  Can also be used
//  to initialize variables which are global to (and therefore can be passed to) other
//  functions in this file.  Called in Mesh constructor.
//======================================================================================

void Mesh::InitUserMeshData(ParameterInput *pin)
{

  // read in some global params (to this file)
 
  pa   = pin->GetOrAddReal("problem","pamb",1.0);
  da   = pin->GetOrAddReal("problem","damb",1.0);
  gamma_gas = pin->GetReal("hydro","gamma");

  Ggrav = pin->GetOrAddReal("problem","Ggrav",6.67408e-8);
  GM2 = pin->GetOrAddReal("problem","M2",1.989e30)*Ggrav;
  GM1 = pin->GetOrAddReal("problem","M1",1.989e33)*Ggrav;

  rsoft2 = pin->GetOrAddReal("problem","rsoft2",0.1);
  corotating_frame = pin->GetInteger("problem","corotating_frame");
    
  p_night = pin->GetReal("problem","p_night");
  rp_aniso_inner = pin->GetReal("problem","rp_aniso_inner");

  trackfile_dt = pin->GetOrAddReal("problem","trackfile_dt",0.01);
  n_particle_substeps = pin->GetInteger("problem","n_particle_substeps");

  //rho_surface_star = pin->GetOrAddReal("problem","rho_surface_star",1.e-15);
  Real mdot_star   = pin->GetReal("problem","mdot_star");
  lambda_star = pin->GetOrAddReal("problem","lambda_star",5.0);
  star_mode   = pin->GetInteger("problem","star_mode");
  

  //rho_surface_planet = pin->GetOrAddReal("problem","rho_surface_planet",1.e-15);
  Real mdot_planet = pin->GetReal("problem","mdot_planet");
  lambda_planet = pin->GetOrAddReal("problem","lambda_planet",5.0);
  radius_planet = pin->GetOrAddReal("problem","radius_planet",6.955e10);
  aniso_heat    = pin->GetOrAddBoolean("problem","aniso_heat",false);
   
  r_inner = pin->GetReal("mesh","x1min");
  //x1_min_derefine = pin->GetOrAddReal("problem","x1_min_derefine",0.0);
  //threshold = pin->GetOrAddReal("problem","ref_grad_thr",0.1);
  //dscale = pin->GetOrAddReal("problem","ref_dscale",1.e10);
  

  // local vars
  Real sma = pin->GetOrAddReal("problem","sma",1.5e12);
  Real ecc = pin->GetOrAddReal("problem","ecc",0.0);
  Real f_corot_planet = pin->GetOrAddReal("problem","f_corotation_planet",1.0);
  Real f_corot_star   = pin->GetOrAddReal("problem","f_corotation_star",1.0);
  Real Omega_orb, vcirc;

  // set star/planet surface density
  // NOTE: this is strictly correct for \gamma=1, will be different for varying gamma
  rho_surface_star = mdot_star / (3.14159*sqrt(GM1*pow(r_inner*lambda_star,3))*exp(1.5-lambda_star) );
  rho_surface_planet = mdot_planet / (3.14159*sqrt(GM2*pow(radius_planet*lambda_planet,3))*exp(1.5-lambda_planet) );
  
  vr_star = 0.0; //mdot_star/(4*3.14159*r_inner*r_inner*rho_surface_star);
  vr_planet = 0.0; //mdot_planet/(4.*3.14159*radius_planet*radius_planet*rho_surface_planet);
  
  // allocate MESH data for the particle pos/vel, Omega frame, omega_planet & omega_star
  AllocateRealUserMeshDataField(4);
  ruser_mesh_data[0].NewAthenaArray(3);
  ruser_mesh_data[1].NewAthenaArray(3);
  ruser_mesh_data[2].NewAthenaArray(3);
  ruser_mesh_data[3].NewAthenaArray(2);

  // enroll the BCs
  if(mesh_bcs[BoundaryFace::outer_x1] == GetBoundaryFlag("user")) {
    EnrollUserBoundaryFunction(BoundaryFace::outer_x1, DiodeOuterX1);
  }
  if(mesh_bcs[BoundaryFace::inner_x1] == GetBoundaryFlag("user")) {
    if(star_mode==1){
      EnrollUserBoundaryFunction(BoundaryFace::inner_x1, AccreteInnerX1);
    }
    if(star_mode==2){
      EnrollUserBoundaryFunction(BoundaryFace::inner_x1, WindInnerX1);
    }
}


  // Enroll a Source Function
  EnrollUserExplicitSourceFunction(StarPlanetWinds);

  // Enroll extra history output
  AllocateUserHistoryOutput(2);
  EnrollUserHistoryOutput(0, MdotMeshPlanet, "md_m_p");
  EnrollUserHistoryOutput(1, MdotMeshStar, "md_m_s");

   // Enroll AMR
  //if(adaptive==true)
  //  EnrollUserRefinementCondition(RefinementCondition);

  // always write at startup
  trackfile_next_time = time;
  trackfile_number = 0;
    
  // PARTICLE ICs
  //ONLY enter ICs loop if this isn't a restart
  if(time==0){
    //Real vcirc = sqrt((GM1+GM2)/sma + accel*sma);    
    vcirc = sqrt((GM1+GM2)/sma);
    Omega_orb = vcirc/sma;
    // rotation of star and planet
    omega_star = f_corot_star * Omega_orb;
    omega_planet = f_corot_planet * Omega_orb;
    
    // set the initial conditions for the pos/vel of the secondary
    xi[0] = -sma*(1.0 + ecc);  // apocenter
    xi[1] = 0.0;
    xi[2] = 0.0;
    
    vi[0] = 0.0;
    vi[1]= -sqrt( vcirc*vcirc*(1.0 - ecc)/(1.0 + ecc) ); //v_apocenter
    vi[2] = 0.0;
    
    // now set the initial condition for Omega
    Omega[0] = 0.0;
    Omega[1] = 0.0;
    Omega[2] = 0.0;
    
    // In the case of a corotating frame,
    // subtract off the frame velocity and set Omega
    if(corotating_frame == 1){
      Omega[2] = Omega_orb;
      vi[1] -=  Omega[2]*xi[0]; 
    }

    // save the ruser_mesh_data variables
    for(int i=0; i<3; i++){
      ruser_mesh_data[0](i)  = xi[i];
      ruser_mesh_data[1](i)  = vi[i];
      ruser_mesh_data[2](i)  = Omega[i];
    }

    ruser_mesh_data[3](0) = omega_planet;
    ruser_mesh_data[3](1) = omega_star;
    
    
  }else{
    is_restart=1;
  }
  
    
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
    std::cout << "e = "<< ecc <<"\n";
    std::cout << "P = "<< 6.2832*sqrt(sma*sma*sma/(GM1+GM2)) << "\n";
    std::cout << "rsoft2 ="<<rsoft2<<"\n";
    std::cout << "corotating frame? = "<< corotating_frame<<"\n";
    std::cout << "particle substeping n="<<n_particle_substeps<<"\n";
    std::cout << "==========================================================\n";
    std::cout << "==========   BC INFO         =============================\n";
    std::cout << "==========================================================\n";
    std::cout << "rho_surface (star) = "<< rho_surface_star <<"\n";
    std::cout << "lambda (star) = "<< lambda_star <<"\n";
    std::cout << "press_surface (star) =" << rho_surface_star*GM1/(r_inner*gamma_gas*lambda_star) <<"\n";
    std::cout << "sound speed surface (star) =" << sqrt(GM1/(r_inner*lambda_star)) <<"\n";
    std::cout << "estimated mdot (star) = "<<3.14159*rho_surface_star*sqrt(GM1*pow(r_inner*lambda_star,3))*exp(1.5-lambda_star) <<"\n";
    std::cout << "rho_surface (planet) = "<< rho_surface_planet <<"\n";
    std::cout << "lambda (planet) = "<< lambda_planet <<"\n";
    std::cout << "aniso_heat (planet) =" << aniso_heat << "\n";
    std::cout << "p_night ="<< p_night <<"\n";
    std::cout << "rp_aniso_inner ="<< rp_aniso_inner <<"\n";
    std::cout << "press_surface (planet) =" << rho_surface_planet*GM2/(radius_planet*gamma_gas*lambda_planet) <<"\n";
    std::cout << "sound speed surface (star) =" << sqrt(GM2/(r_inner*lambda_planet)) <<"\n";
    std::cout << "estimated mdot (planet) = "<<3.14159*rho_surface_planet*sqrt(GM2*pow(radius_planet*lambda_planet,3))*exp(1.5-lambda_planet) <<"\n";
    std::cout << "==========================================================\n";
    std::cout << "==========   Particle        =============================\n";
    std::cout << "==========================================================\n";
    std::cout << "x ="<<xi[0]<<"\n";
    std::cout << "y ="<<xi[1]<<"\n";
    std::cout << "z ="<<xi[2]<<"\n";
    std::cout << "vx ="<<vi[0]<<"\n";
    std::cout << "vy ="<<vi[1]<<"\n";
    std::cout << "vz ="<<vi[2]<<"\n";
    std::cout << "==========================================================\n";
  }
  
} // end




Real press_factor_aniso(Real d2, Real solar_angle)
{
  Real press_factor = 1.0;
  Real press_factor_aniso = 1.0;
  Real amplitude_scale = (1-p_night)/2;
  Real linear_slope_radial = 1.0;
  if (aniso_heat){
    // defines the pressure scaling according to solar angle;
    // press_factor = 1 at day-side (0-90 and 270-360 deg);
    if (solar_angle<=1.57080){
        press_factor_aniso = amplitude_scale * std::cos(2 * solar_angle) + amplitude_scale + p_night;
      }else if(solar_angle>=4.71239){
        press_factor_aniso = amplitude_scale * std::cos(2 * solar_angle) + amplitude_scale + p_night;
    //press_factor = p_night at night-side.
      }else{
        press_factor_aniso = p_night;
      }
    // defines the pressure scaling according to radial distance to planet center;
    // press_factor = 1 for the very inner region; if you change the factor before radius_planet,
      if(d2<=rp_aniso_inner){
        press_factor = 1.0;
    // linear scaling from 1 (at center) to press_factor_aniso (at r = Rp)
      }else if(d2<=radius_planet){
        linear_slope_radial = -(1 - press_factor_aniso) / (radius_planet - rp_aniso_inner);
        press_factor = linear_slope_radial * d2 + 1 - linear_slope_radial * rp_aniso_inner;
      }else{
        press_factor = 1.0;
      }
  }
  return press_factor;
}



//======================================================================================
//! \fn void MeshBlock::ProblemGenerator(ParameterInput *pin)
//  \brief Spherical Coords HSE Envelope problem generator
//======================================================================================

void MeshBlock::ProblemGenerator(ParameterInput *pin)
{

  // local vars
  Real den, pres, vr;
  Real sma = pin->GetReal("problem","sma");
  
  // Prepare index bounds including ghost cells
  int il = is - NGHOST;
  int iu = ie + NGHOST;
  int jl = js;
  int ju = je;
  if (block_size.nx2 > 1) {
    jl -= (NGHOST);
    ju += (NGHOST);
  }
  int kl = ks;
  int ku = ke;
  if (block_size.nx3 > 1) {
    kl -= (NGHOST);
    ku += (NGHOST);
  }
  
  // SETUP THE INITIAL CONDITIONS ON MESH
  for (int k=kl; k<=ku; k++) {
    for (int j=jl; j<=ju; j++) {
      for (int i=il; i<=iu; i++) {

	Real r  = pcoord->x1v(i);
	Real th = pcoord->x2v(j);
	Real ph = pcoord->x3v(k);

	Real sin_th = sin(th);
	Real Rcyl = r*sin_th;

	// INITIALIZE STAR BOUNDARY AND STELLAR WIND
	// current position of the secondary
	Real x_2 = xi[0];
	Real y_2 = xi[1];
	Real z_2 = xi[2];
	
	// spherical polar coordinates, get local cartesian           
	Real x = r*sin(th)*cos(ph);
	Real y = r*sin(th)*sin(ph);
	Real z = r*cos(th);

	// location relative to point mass 2 (star)
	Real d2  = sqrt(pow(x-xi[0], 2) +
			pow(y-xi[1], 2) +
			pow(z-xi[2], 2) );
	Real R2 =  sqrt(pow(x-xi[0], 2) +
			pow(y-xi[1], 2) );
	Real phi2 = std::atan2(y-xi[1],x-xi[0]);
	if (phi2 < 0){
	  phi2 += 2*3.1415926535; 
	}
	Real th2 = std::acos((z-xi[2])/d2);
	Real solar_angle = std::acos( std::sin(th2)*std::cos( - phi2) );
	Real press_factor = press_factor_aniso(d2,solar_angle);

	// surface parameters (star and planet)
	Real press_surface_planet = rho_surface_planet*GM2/(radius_planet*gamma_gas*lambda_planet);
	Real cs_planet = std::sqrt(gamma_gas *press_surface_planet/rho_surface_planet);
	Real press_surface_star = rho_surface_star*GM1/(r*gamma_gas*lambda_star);
	Real cs_star = std::sqrt(gamma_gas *press_surface_star/rho_surface_star);
	
	Real vx,vy,vz;
	Real vr,vth,vph;

	

	// Near Planet
	if(d2 <= radius_planet){
	  den = rho_surface_planet;
	  pres = press_factor*press_surface_planet;
	  vx = vi[0] - sin(phi2)*(omega_planet-Omega[2])*R2;
	  vy = vi[1] + cos(phi2)*(omega_planet-Omega[2])*R2;
	  vz = vi[2];
	  vr  = sin(th)*cos(ph)*vx + sin(th)*sin(ph)*vy + cos(th)*vz;
	  vth = cos(th)*cos(ph)*vx + cos(th)*sin(ph)*vy - sin(th)*vz;
	  vph = -sin(ph)*vx + cos(ph)*vy;
	}else if(d2<= sma/2){
	  den = rho_surface_planet * pow((d2/radius_planet),-8);
	  pres = press_surface_planet * pow(den / rho_surface_planet, gamma_gas);
	  // wind directed outward at v=cs
	  // constant angular momentum of surface
	  //vx = (x-xi[0])/d2 * cs_planet + vi[0] - sin(phi2)*(omega_planet*radius_planet*radius_planet*std::sin(th2)*std::sin(th2)/R2 - Omega[2]*R2);  
	  //vy = (y-xi[1])/d2 * cs_planet + vi[1] + cos(phi2)*(omega_planet*radius_planet*radius_planet*std::sin(th2)*std::sin(th2)/R2 - Omega[2]*R2);  
	  //vz = (z-xi[2])/d2 * cs_planet + vi[2];
	  vx = vi[0];
	  vy = vi[1];
	  vz = vi[2];
	  vr  = sin(th)*cos(ph)*vx + sin(th)*sin(ph)*vy + cos(th)*vz;
	  vth = cos(th)*cos(ph)*vx + cos(th)*sin(ph)*vy - sin(th)*vz;
	  vph = -sin(ph)*vx + cos(ph)*vy;
	}else{
	  den = da;
	  pres = pa;
	  vr = 0.0;
	  vth = 0.0;
	  vph = 0.0; //- Omega[2]*Rcyl;
	}
	
	/*
	// Near star, if there is a wind BC
	if((r<sma/2) && (star_mode==2)  ){
	  // wind directed outward at v=cs outside of sonic point, linear increase to sonic point
	  // constant angular momentum of surface
	  den = rho_surface_star * pow((r/r_inner),-8);
	  pres = press_surface_star * pow(den / rho_surface_star, gamma_gas);
	  vr = sqrt(2*GM1/r); //cs_star; // * std::min(r/(lambda_star/2. * r_inner), 1.0);  
	  vth = 0.0;
	  vph = omega_star*SQR(r_inner*sin_th)/Rcyl - Omega[2]*Rcyl;
	}
	*/
	
	
	phydro->u(IDN,k,j,i) = std::max(den,da);
	phydro->u(IM1,k,j,i) = den*vr;
	phydro->u(IM2,k,j,i) = den*vth;
	phydro->u(IM3,k,j,i) = den*vph;
	phydro->u(IEN,k,j,i) = std::max(pres,pa)/(gamma_gas-1.0);
	phydro->u(IEN,k,j,i) += 0.5*(SQR(phydro->u(IM1,k,j,i))+SQR(phydro->u(IM2,k,j,i))
				     + SQR(phydro->u(IM3,k,j,i)))/phydro->u(IDN,k,j,i);
      }
    }
  } // end loop over cells
  return;
} // end ProblemGenerator






// Source Function for two point masses
void StarPlanetWinds(MeshBlock *pmb, const Real time, const Real dt,  const AthenaArray<Real> *flux,
		  const AthenaArray<Real> &prim,
		  const AthenaArray<Real> &prim_scalar, const AthenaArray<Real> &bcc,
		  AthenaArray<Real> &cons, AthenaArray<Real> &cons_scalar)
{ 

  if(is_restart>0){
    // else this is a restart, read the current particle state
    for(int i=0; i<3; i++){
      xi[i]    = pmb->pmy_mesh->ruser_mesh_data[0](i);
      vi[i]    = pmb->pmy_mesh->ruser_mesh_data[1](i);
      Omega[i] = pmb->pmy_mesh->ruser_mesh_data[2](i);
    }
    omega_planet = pmb->pmy_mesh->ruser_mesh_data[3](0);
    omega_star   = pmb->pmy_mesh->ruser_mesh_data[3](1);

    // print some info
    if (Globals::my_rank==0){
      std::cout << "*** Setting initial conditions for t>0 ***\n";
      std::cout <<"xi="<<xi[0]<<" "<<xi[1]<<" "<<xi[2]<<"\n";
      std::cout <<"vi="<<vi[0]<<" "<<vi[1]<<" "<<vi[2]<<"\n";
      std::cout <<"Omega="<<Omega[0]<<" "<<Omega[1]<<" "<<Omega[2]<<"\n";
      std::cout << "omega_planet ="<<omega_planet<<"  omega_star ="<<omega_star<<"\n";
    }
    is_restart=0;
  }
  
  // Gravitational acceleration from orbital motion
  for (int k=pmb->ks; k<=pmb->ke; k++) {
    Real ph= pmb->pcoord->x3v(k);
    Real sin_ph = sin(ph);
    Real cos_ph = cos(ph);
    for (int j=pmb->js; j<=pmb->je; j++) {
      Real th= pmb->pcoord->x2v(j);
      Real sin_th = sin(th);
      Real cos_th = cos(th);
      for (int i=pmb->is; i<=pmb->ie; i++) {
	Real r = pmb->pcoord->x1v(i);
	
	// current position of the secondary
	Real x_2 = xi[0];
	Real y_2 = xi[1];
	Real z_2 = xi[2];
	Real d12c = pow(xi[0]*xi[0] + xi[1]*xi[1] + xi[2]*xi[2], 1.5);
	
	// spherical polar coordinates, get local cartesian           
	Real x = r*sin_th*cos_ph;
	Real y = r*sin_th*sin_ph;
	Real z = r*cos_th;
  
	Real d2  = sqrt(pow(x-x_2, 2) +
			pow(y-y_2, 2) +
			pow(z-z_2, 2) );
  
	//
	//  COMPUTE ACCELERATIONS 
	//
	// PM1
	Real a_r1 = -GM1*pmb->pcoord->coord_src1_i_(i)/r;
	
	// PM2 gravitational accels in cartesian coordinates
	Real a_x = - GM2 * fspline(d2,rsoft2) * (x-x_2);   
	Real a_y = - GM2 * fspline(d2,rsoft2) * (y-y_2);  
	Real a_z = - GM2 * fspline(d2,rsoft2) * (z-z_2);
	
	// add the correction for the orbiting frame (relative to the COM)
	a_x += -  GM2 / d12c * x_2;
	a_y += -  GM2 / d12c * y_2;
	a_z += -  GM2 / d12c * z_2;
	
	if(corotating_frame == 1){
	  
	  Real vr  = prim(IVX,k,j,i);
	  Real vth = prim(IVY,k,j,i);
	  Real vph = prim(IVZ,k,j,i);
	  
	  // distance from the origin in cartesian (vector)
	  Real rxyz[3];
	  rxyz[0] = x;
	  rxyz[1] = y;
	  rxyz[2] = z;
	  
	  // get the cartesian velocities from the spherical (vector)
	  Real vgas[3];
	  vgas[0] = sin_th*cos_ph*vr + cos_th*cos_ph*vth - sin_ph*vph;
	  vgas[1] = sin_th*sin_ph*vr + cos_th*sin_ph*vth + cos_ph*vph;
	  vgas[2] = cos_th*vr - sin_th*vth;
	  
	  // add the centrifugal and coriolis terms
	  
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
	}
	
	// convert back to spherical
	Real a_r  = sin_th*cos_ph*a_x + sin_th*sin_ph*a_y + cos_th*a_z;
	Real a_th = cos_th*cos_ph*a_x + cos_th*sin_ph*a_y - sin_th*a_z;
	Real a_ph = -sin_ph*a_x + cos_ph*a_y;
	
	// add the PM1 accel
	a_r += a_r1;
	
	//
	// ADD SOURCE TERMS TO THE GAS MOMENTA/ENERGY
	//
	Real den = prim(IDN,k,j,i);
	
	Real src_1 = dt*den*a_r; 
	Real src_2 = dt*den*a_th;
	Real src_3 = dt*den*a_ph;
	
	// add the source term to the momenta  (source = - rho * a)
	cons(IM1,k,j,i) += src_1;
	cons(IM2,k,j,i) += src_2;
	cons(IM3,k,j,i) += src_3;
	
	// update the energy (source = - rho v dot a)
	cons(IEN,k,j,i) += src_1/den * 0.5*(flux[X1DIR](IDN,k,j,i) + flux[X1DIR](IDN,k,j,i+1));
	cons(IEN,k,j,i) += src_2*prim(IVY,k,j,i) + src_3*prim(IVZ,k,j,i);

      }
    }
  } // end loop over cells
  

}







Real MdotMeshPlanet(MeshBlock *pmb, int iout){
  // mass flux on/off of the mesh, >0 = inflow, <0 = outflow
  Real mdot = 0.0;
  
  if(pmb->pbval->block_bcs[BoundaryFace::inner_x1] == BoundaryFlag::user) {
     int is=pmb->is, ie=pmb->ie, js=pmb->js, je=pmb->je, ks=pmb->ks, ke=pmb->ke;
     AthenaArray<Real> area;
     int ncells1 = pmb->block_size.nx1 + 2*(NGHOST);
     area.NewAthenaArray(ncells1);
  
     for(int k=ks; k<=ke; k++) {
       for(int j=js; j<=je; j++) {
          pmb->pcoord->VolCenterFace1Area(k,j,is,ie,area);
          mdot += area(is)*pmb->phydro->u(IM1,k,j,is)*pmb->pscalars->r(0,k,j,is); // dmdot = dA*rho*v
        }
      }
    } // end if

  if(pmb->pbval->block_bcs[BoundaryFace::outer_x1] == BoundaryFlag::user) {
     int is=pmb->is, ie=pmb->ie, js=pmb->js, je=pmb->je, ks=pmb->ks, ke=pmb->ke;
     AthenaArray<Real> area;
     int ncells1 = pmb->block_size.nx1 + 2*(NGHOST);
     area.NewAthenaArray(ncells1);
  
     for(int k=ks; k<=ke; k++) {
       for(int j=js; j<=je; j++) {
          pmb->pcoord->VolCenterFace1Area(k,j,is,ie,area);
          mdot += -area(ie)*pmb->phydro->u(IM1,k,j,ie)*pmb->pscalars->r(0,k,j,ie); // dmdot = dA*rho*v
        }
      }
    } // end if

  
  return mdot;
}


Real MdotMeshStar(MeshBlock *pmb, int iout){
  // mass flux on/off of the mesh, >0 = inflow, <0 = outflow
  Real mdot = 0.0;
  
  if(pmb->pbval->block_bcs[BoundaryFace::outer_x1] == BoundaryFlag::user) {
     int is=pmb->is, ie=pmb->ie, js=pmb->js, je=pmb->je, ks=pmb->ks, ke=pmb->ke;
     AthenaArray<Real> area;
     int ncells1 = pmb->block_size.nx1 + 2*(NGHOST);
     area.NewAthenaArray(ncells1);
  
     for(int k=ks; k<=ke; k++) {
       for(int j=js; j<=je; j++) {
          pmb->pcoord->VolCenterFace1Area(k,j,is,ie,area);
          mdot += -area(ie)*pmb->phydro->u(IM1,k,j,ie)*pmb->pscalars->r(1,k,j,ie); // dmdot = dA*rho*v
        }
      }
    } // end if

  
  return mdot;
}



// int RefinementCondition(MeshBlock *pmb)
// {

//   int mylevel = pmb->loc.level;
      
//   int refine = 0;
//   Real mindist=1.e99;
//   Real rmin = 1.e99;

//   AthenaArray<Real> &w = pmb->phydro->w;
//   Real maxeps = 0.0;

//   Real maxscalar;

//   for(int k=pmb->ks; k<=pmb->ke; k++){
//     Real ph= pmb->pcoord->x3v(k);
//     Real sin_ph = sin(ph);
//     Real cos_ph = cos(ph);
//     for(int j=pmb->js; j<=pmb->je+1; j++) {
//       Real th= pmb->pcoord->x2v(j);
//       Real sin_th = sin(th);
//       Real cos_th = cos(th);
//       for(int i=pmb->is; i<=pmb->ie; i++) {
// 	// distance from the planet
// 	Real r = pmb->pcoord->x1v(i);
// 	Real x = r*sin_th*cos_ph;
// 	Real y = r*sin_th*sin_ph;
// 	Real z = r*cos_th;
// 	Real dist = std::sqrt(SQR(x-xi[0]) +
// 			      SQR(y-xi[1]) +
// 			      SQR(z-xi[2]) );
// 	mindist = std::min(mindist,dist);
// 	rmin    = std::min(rmin,r);

// 	// pressure gradients
// 	Real eps = std::sqrt(SQR(0.5*(w(IPR,k,j,i+1) - w(IPR,k,j,i-1)))
// 			     +SQR(0.5*(w(IPR,k,j+1,i) - w(IPR,k,j-1,i)))
// 			     +SQR(0.5*(w(IPR,k+1,j,i) - w(IPR,k-1,j,i))))/w(IPR,k,j,i);
// 	maxeps = std::max(maxeps, eps);

// 	// scalar concentration
// 	Real scalar_r = pmb->pscalars->r(0,k,j,i);
// 	maxscalar = std::max(maxscalar,scalar_r);


//       }
//     }
//   }

//   int lrefinemax = pmb->pmy_mesh->max_level; //- floor( std::max(0.0,log2(mindist/dscale)) );
  
  
//   // based on distance from the planet
//   if(mindist <= 6.0*radius_planet){
//     refine = 1;
//   } else if(mindist > 6.0*radius_planet) {
//     // based on threshold of pw scalar
//     if(maxscalar>0.01){
//       if ( (maxeps > threshold) && (mylevel < lrefinemax) ){
// 	refine = 1;
//       } else if (maxeps < 0.25*threshold){
// 	refine = -1;
//       } else {
// 	refine = 0;
//       }
//     }else{
//       refine = -1;
//     }    
//   }

//   // Override AMR on the polar boundaries (because we turned off a safety check
//   if(pmb->pbval->block_bcs[BoundaryFace::inner_x2] == BoundaryFlag::polar ||
//      pmb->pbval->block_bcs[BoundaryFace::outer_x2] == BoundaryFlag::polar) {
//     refine = 0;
//   }
  

//   return refine;
//   //return 0;
// }




void MeshBlock::UserWorkInLoop(void) {
  // Add timestep diagnostics
  if(pmy_mesh->ncycle % 10 == 0){
    if(new_block_dt_ == pmy_mesh->dt){
      // call NewBlockTimeStep with extra diagnostic output
      phydro->NewBlockTimeStep(1);
    }
  }  


  for (int k=ks; k<=ke; k++) {
    Real ph=pcoord->x3v(k);
    Real sin_ph = sin(ph);
    Real cos_ph = cos(ph);
    for (int j=js; j<=je; j++) {
      Real th=pcoord->x2v(j);
      Real sin_th = sin(th);
      Real cos_th = cos(th);
      for (int i=is; i<=ie; i++) {
	Real r =pcoord->x1v(i);
	
	// spherical polar coordinates, get local cartesian           
	Real x = r*sin_th*cos_ph;
	Real y = r*sin_th*sin_ph;
	Real z = r*cos_th;
  
	Real d2  = sqrt(pow(x-xi[0], 2) +
			pow(y-xi[1], 2) +
			pow(z-xi[2], 2) );

	// PLANET BOUNDARY (note, overwrites the grav accel, ie gravitational accel is not applied in this region)
	if(d2 <= radius_planet){
	  Real R2 =  sqrt(pow(x-xi[0], 2) +
			  pow(y-xi[1], 2) );
	  Real phi2 = std::atan2(y-xi[1],x-xi[0]);
	  if (phi2 < 0){ 
	    phi2 += 2*3.1415926535; 
	  }
	  Real th2 = std::acos((z-xi[2])/d2);
	  Real solar_angle = std::acos( std::sin(th2)*std::cos( - phi2) );
	  Real press_factor = press_factor_aniso(d2,solar_angle);
	 
	  Real press_surface_planet = rho_surface_planet*GM2/(radius_planet*gamma_gas*lambda_planet);
	  Real cs = std::sqrt(gamma_gas *press_surface_planet/rho_surface_planet);
	  Real vx = vi[0] - sin(phi2)*(omega_planet-Omega[2])*R2 + vr_planet*(x-xi[0])/d2;
	  Real vy = vi[1] + cos(phi2)*(omega_planet-Omega[2])*R2 + vr_planet*(y-xi[1])/d2;
	  Real vz = vi[2] + vr_planet*(z-xi[2])/d2;

	  // convert back to spherical polar
	  Real vr  = sin_th*cos_ph*vx + sin_th*sin_ph*vy + cos_th*vz;
	  Real vth = cos_th*cos_ph*vx + cos_th*sin_ph*vy - sin_th*vz;
	  Real vph = -sin_ph*vx + cos_ph*vy;
	  
	  phydro->u(IDN,k,j,i) = rho_surface_planet;
	  phydro->u(IM1,k,j,i) = rho_surface_planet*vr;
	  phydro->u(IM2,k,j,i) = rho_surface_planet*vth;  
	  phydro->u(IM3,k,j,i) = rho_surface_planet*vph;  
	  phydro->u(IEN,k,j,i) = press_factor*press_surface_planet/(gamma_gas-1.0);
	  phydro->u(IEN,k,j,i) += 0.5*(SQR(phydro->u(IM1,k,j,i))+SQR(phydro->u(IM2,k,j,i))
				       + SQR(phydro->u(IM3,k,j,i)))/phydro->u(IDN,k,j,i);
	  
	  pscalars->s(0,k,j,i) = 1.0*rho_surface_planet; // set scalar concetration to one
	  pscalars->s(1,k,j,i) = scalar_val*rho_surface_planet; 
	} // within planet radius

	/*
	// STAR BOUNDARY(note, overwrites the grav accel, ie gravitational accel is not applied in this region)
	if(pbval->block_bcs[BoundaryFace::inner_x1] == BoundaryFlag::reflect) {
	  Real Rcyl = r*sin(th);
	  Real press_surface_star = rho_surface_star*GM1/(r*gamma_gas*lambda_star);
	  phydro->u(IDN,k,j,is) = rho_surface_star;
	  phydro->u(IM1,k,j,is) = 0.0;
	  phydro->u(IM2,k,j,is) = 0.0;
	  phydro->u(IM3,k,j,is) = rho_surface_star*(omega_star-Omega[2])*Rcyl;
	  phydro->u(IEN,k,j,is) = press_surface_star/(gamma_gas-1.0);
	  phydro->u(IEN,k,j,i) += 0.5*(SQR(phydro->u(IM1,k,j,i))+SQR(phydro->u(IM2,k,j,i))
					  + SQR(phydro->u(IM3,k,j,i)))/phydro->u(IDN,k,j,i);
	}// within star bc layer
	*/

      }
    }
  } // end loop over zones


  return;
}











//========================================================================================
// MM
//! \fn void MeshBlock::MeshUserWorkInLoop(void)
//  \brief Function called once every time step for user-defined work.
//========================================================================================

void Mesh::MeshUserWorkInLoop(ParameterInput *pin){

  Real ai[3];
  Real mg;
  Mesh *pm = my_blocks(0)->pmy_mesh;
  
  // kick the initial conditions back a half step (v^n-1/2)
  if(ncycle==0){
    ParticleAccels(xi,vi,ai);
    kick(-0.5*dt,xi,vi,ai); 
  } // ncycle=0 

    
  // EVOLVE THE ORBITAL POSITION OF THE SECONDARY
  // do this on rank zero, then broadcast
  if (Globals::my_rank == 0){
    for (int ii=1; ii<=n_particle_substeps; ii++) {
      // add the particle acceleration to ai
      ParticleAccels(xi,vi,ai);
      // advance the particle
      particle_step(dt/n_particle_substeps,xi,vi,ai);
    }
  }
  
#ifdef MPI_PARALLEL
  // broadcast the position update from proc zero
  MPI_Bcast(xi,3,MPI_ATHENA_REAL,0,MPI_COMM_WORLD);
  MPI_Bcast(vi,3,MPI_ATHENA_REAL,0,MPI_COMM_WORLD);
#endif

  // update the ruser_mesh_data variables
  for(int i=0; i<3; i++){
    ruser_mesh_data[0](i)  = xi[i];
    ruser_mesh_data[1](i)  = vi[i];
    ruser_mesh_data[2](i)  = Omega[i];
  }
  
  // write the output to the trackfile
  if(time >= trackfile_next_time || user_force_output ){
    WritePMTrackfile(pm,pin);
  }
  
}


void WritePMTrackfile(Mesh *pm, ParameterInput *pin){
  
  if (Globals::my_rank == 0) {
    std::string fname;
    fname.assign("pm_trackfile.dat");
    
    // open file for output
    FILE *pfile;
    std::stringstream msg;
    if((pfile = fopen(fname.c_str(),"a")) == NULL){
      msg << "### FATAL ERROR in function [WritePMTrackfile]" << std::endl
          << "Output file '" << fname << "' could not be opened";
      throw std::runtime_error(msg.str().c_str());
    }
  
    if(trackfile_number==0){
      fprintf(pfile,"#    ncycle     ");
      fprintf(pfile,"time           ");
      fprintf(pfile,"dt             ");
      fprintf(pfile,"m1             ");
      fprintf(pfile,"m2             ");
      fprintf(pfile,"x              ");
      fprintf(pfile,"y              ");
      fprintf(pfile,"z              ");
      fprintf(pfile,"vx             ");
      fprintf(pfile,"vy             ");
      fprintf(pfile,"vz             ");
      fprintf(pfile,"\n");
    }


    // write the data line
    fprintf(pfile,"%20i",pm->ncycle);
    fprintf(pfile,"%20.6e",pm->time);
    fprintf(pfile,"%20.6e",pm->dt);
    fprintf(pfile,"%20.6e",GM1/Ggrav);
    fprintf(pfile,"%20.6e",GM2/Ggrav);
    fprintf(pfile,"%20.6e",xi[0]);
    fprintf(pfile,"%20.6e",xi[1]);
    fprintf(pfile,"%20.6e",xi[2]);
    fprintf(pfile,"%20.6e",vi[0]);
    fprintf(pfile,"%20.6e",vi[1]);
    fprintf(pfile,"%20.6e",vi[2]);
    fprintf(pfile,"\n");

    // close the file
    fclose(pfile);  

  } // end rank==0

  // increment counters
  trackfile_number++;
  trackfile_next_time += trackfile_dt;
  
  return;
}


//--------------------------------------------------------------------------------------
//! \fn void OutflowOuterX1(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
//                         FaceField &b, Real time, Real dt,
//                         int is, int ie, int js, int je, int ks, int ke)
//  \brief OUTFLOW boundary conditions, outer x1 boundary

void DiodeOuterX1(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
		    FaceField &b, Real time, Real dt, int is, int ie, int js, int je, int ks, int ke, int ngh)
{
  // copy hydro variables into ghost zones, don't allow inflow
  for (int n=0; n<(NHYDRO); ++n) {
    if (n==(IVX)) {
      for (int k=ks; k<=ke; ++k) {
	for (int j=js; j<=je; ++j) {
#pragma simd
	  for (int i=1; i<=(NGHOST); ++i) {
	    prim(IVX,k,j,ie+i) =  std::max( 0.0, prim(IVX,k,j,(ie-i+1)) );  // positive velocities only
	  }
	}}
    } else {
      for (int k=ks; k<=ke; ++k) {
	for (int j=js; j<=je; ++j) {
#pragma simd
	  for (int i=1; i<=(NGHOST); ++i) {
	    prim(n,k,j,ie+i) = prim(n,k,j,(ie-i+1));
	  }
	}}
    }
  }


  /*
  // copy face-centered magnetic fields into ghost zones
  if (MAGNETIC_FIELDS_ENABLED) {
    for (int k=ks; k<=ke; ++k) {
      for (int j=js; j<=je; ++j) {
#pragma simd
	for (int i=1; i<=(NGHOST); ++i) {
	  b.x1f(k,j,(ie+i+1)) = b.x1f(k,j,(ie+1));
	}
      }}

    for (int k=ks; k<=ke; ++k) {
      for (int j=js; j<=je+1; ++j) {
#pragma simd
	for (int i=1; i<=(NGHOST); ++i) {
	  b.x2f(k,j,(ie+i)) = b.x2f(k,j,ie);
	}
      }}

    for (int k=ks; k<=ke+1; ++k) {
      for (int j=js; j<=je; ++j) {
#pragma simd
	for (int i=1; i<=(NGHOST); ++i) {
	  b.x3f(k,j,(ie+i)) = b.x3f(k,j,ie);
	}
      }}
  }
  */

  return;
}


//----------------------------------------------------------------------------------------
//! \fn void WindInnerX1()
void WindInnerX1(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim, FaceField &b,
                Real time, Real dt,
                int il, int iu, int jl, int ju, int kl, int ku, int ngh) {
  for (int k=kl; k<=ku; ++k) {
    for (int j=jl; j<=ju; ++j) {
      Real sin_th = sin(pco->x2v(j));
      for (int i=1; i<=ngh; ++i) {
	Real r = pco->x1v(il-i);
	prim(IDN,k,j,il-i) = rho_surface_star;
	prim(IVX,k,j,il-i) = vr_star;
	prim(IVY,k,j,il-i) = 0.0;
	prim(IVZ,k,j,il-i) = r*sin_th*(omega_star-Omega[2]);
	prim(IPR,k,j,il-i) = rho_surface_star*GM1/(r*gamma_gas*lambda_star);
	pmb->pscalars->r(0,k,j,il-i) = scalar_val;
	pmb->pscalars->r(1,k,j,il-i) = 1.0;
      }
    }
  }
}


//----------------------------------------------------------------------------------------
//! \fn void WindInnerX1()
void AccreteInnerX1(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim, FaceField &b,
                Real time, Real dt,
                int il, int iu, int jl, int ju, int kl, int ku, int ngh) {
  
  for (int k=kl; k<=ku; ++k) {
    for (int j=jl; j<=ju; ++j) {
      for (int i=1; i<=ngh; ++i) {
	prim(IDN,k,j,il-i) = 0.1*prim(IDN,k,j,il);
	prim(IVX,k,j,il-i) = std::min( prim(IVX,k,j,il), 0.0);
	prim(IVY,k,j,il-i) = prim(IVY,k,j,il);
	prim(IVZ,k,j,il-i) = prim(IVZ,k,j,il);
	prim(IPR,k,j,il-i) = 0.3*prim(IPR,k,j,il);
	//pscalars->s(0,k,j,il-i) = scalar_val*phydro->u(IDN,k,j,i);
      }
    }
  }

}




  
  

void particle_step(Real dt,Real (&xi)[3],Real (&vi)[3],Real (&ai)[3]){
  // Leapfrog algorithm (KDK)

  // kick a full step
  kick(dt,xi,vi,ai);

  // drift a full step
  drift(dt,xi,vi,ai);
  
}

// kick the velocities dt using the accelerations given in ai
void kick(Real dt,Real (&xi)[3],Real (&vi)[3],Real (&ai)[3]){
  for (int i = 0; i < 3; i++){
    vi[i] += dt*ai[i];
  }
}

// drift the velocities dt using the velocities given in vi
void drift(Real dt,Real (&xi)[3],Real (&vi)[3],Real (&ai)[3]){
  for (int i = 0; i < 3; i++){
    xi[i] += dt*vi[i];
  }
}

void ParticleAccels(Real (&xi)[3],Real (&vi)[3],Real (&ai)[3]){

  Real d = sqrt(xi[0]*xi[0] + xi[1]*xi[1] + xi[2]*xi[2]);

  // fill in the accelerations for the orbiting frame
  for (int i = 0; i < 3; i++){
    ai[i] = - GM1/pow(d,3) * xi[i] - GM2/pow(d,3) * xi[i];
  } 
  
  // IF WE'RE IN A ROTATING FRAME
  if(corotating_frame == 1){
    Real Omega_x_r[3],Omega_x_Omega_x_r[3], Omega_x_v[3];
 
    // compute cross products 
    cross(Omega,xi,Omega_x_r);
    cross(Omega,Omega_x_r,Omega_x_Omega_x_r);
    
    cross(Omega,vi,Omega_x_v);
  
    // fill in the accelerations for the rotating frame
    for (int i = 0; i < 3; i++){
      ai[i] += -Omega_x_Omega_x_r[i];
      ai[i] += -2.0*Omega_x_v[i];
    }
  }
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

