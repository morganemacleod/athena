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
//! \file relax_envelope.cpp: polytropic stellar envelope, monopole self gravity
//configure:  python configure.py --prob relax_envelope --coord spherical_polar  -hdf5 --hdf5_path /opt/local -mpi
//======================================================================================

// C++ headers
#include <sstream>
#include <cmath>
#include <stdexcept>
#include <fstream>
#include <iostream>
#define NARRAY 10000
#define NGRAV1DPROF 200


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




Real Interpolate1DArrayEven(Real *x,Real *y,Real x0, int length);

void DiodeOuterX1(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,FaceField &b,
		 Real time, Real dt, int is, int ie, int js, int je, int ks, int ke, int ngh);

void MyPointMass(MeshBlock *pmb, const Real time, const Real dt,  const AthenaArray<Real> *flux,
		  const AthenaArray<Real> &prim,
		  const AthenaArray<Real> &prim_scalar, const AthenaArray<Real> &bcc,
		  AthenaArray<Real> &cons, AthenaArray<Real> &cons_scalar); 

void SumMencProfile(Mesh *pm, Real (&menc)[NGRAV1DPROF]);

void updateGM2(Real sep);


// global (to this file) problem parameters
Real gamma_gas; 
Real da,pa; // ambient density, pressure
Real rho[NARRAY], p[NARRAY], rad[NARRAY], menc_init[NARRAY];  // initial profile
Real logr[NGRAV1DPROF],menc[NGRAV1DPROF]; // enclosed mass profile

Real GM1,GM1i; // point masses
Real t_relax,t_mass_on; // time to damp fluid motion, time to turn on M2 over
Real Ggrav;


int update_grav_every;
Real tau_relax_start,tau_relax_end;
Real rstar_initial,mstar_initial;


//======================================================================================
//! \fn void Mesh::InitUserMeshData(ParameterInput *pin)
//  \brief Function to initialize problem-specific data in mesh class.  Can also be used
//  to initialize variables which are global to (and therefore can be passed to) other
//  functions in this file.  Called in Mesh constructor.
//======================================================================================

void Mesh::InitUserMeshData(ParameterInput *pin)
{

  // read in some global params (to this file)
 
  // first non-mode-dependent settings
  pa   = pin->GetOrAddReal("problem","pamb",1.0);
  da   = pin->GetOrAddReal("problem","damb",1.0);
  gamma_gas = pin->GetReal("hydro","gamma");

  Ggrav = pin->GetOrAddReal("problem","Ggrav",6.67408e-8);
   
  t_relax = pin->GetOrAddReal("problem","trelax",0.0);
  tau_relax_start = pin->GetOrAddReal("problem","tau_relax_start",1.0);
  tau_relax_end = pin->GetOrAddReal("problem","tau_relax_end",100.0);
  
  // gravity
  update_grav_every = pin->GetOrAddInteger("problem","update_grav_every",1);
  rstar_initial = pin->GetReal("problem","rstar_initial");  // FOR RESCALING OF STELLAR PROFILE
  mstar_initial = pin->GetReal("problem","mstar_initial");
  
  // local vars
  Real rmin = pin->GetOrAddReal("mesh","x1min",0.0);
  Real rmax = pin->GetOrAddReal("mesh","x1max",0.0);
  Real thmin = pin->GetOrAddReal("mesh","x2min",0.0);
  Real thmax = pin->GetOrAddReal("mesh","x2max",0.0);
  
  
  // enroll the BCs
  if(mesh_bcs[BoundaryFace::outer_x1] == GetBoundaryFlag("user")) {
    EnrollUserBoundaryFunction(BoundaryFace::outer_x1, DiodeOuterX1);
  }

  // Enroll a Source Function
  EnrollUserExplicitSourceFunction(MyPointMass);
    
  // read in profile arrays from file
  std::ifstream infile("hse_profile.dat"); 
  for(int i=0;i<NARRAY;i++){
    infile >> rad[i] >> rho[i] >> p[i] >> menc_init[i];
    //std:: cout << rad[i] << "    " << rho[i] << std::endl;
  }
  infile.close();

  // RESCALE
  for(int i=0;i<NARRAY;i++){
    rad[i] = rad[i]*rstar_initial;
    rho[i] = rho[i]*mstar_initial/pow(rstar_initial,3);
    p[i]   = p[i]*Ggrav*pow(mstar_initial,2)/pow(rstar_initial,4);
    menc_init[i] = menc_init[i]*mstar_initial;
  }

  
  
  // set the inner point mass based on excised mass
  Real menc_rin = Interpolate1DArrayEven(rad,menc_init, rmin, NARRAY );
  GM1 = Ggrav*menc_rin;
  GM1i = GM1;
  Real GMenv = Ggrav*Interpolate1DArrayEven(rad,menc_init,1.01*rstar_initial, NARRAY) - GM1;

  // allocate the enclosed mass profile
  Real logr_min = log10(rmin);
  Real logr_max = log10(rmax);
  
  for(int i=0;i<NGRAV1DPROF;i++){
    logr[i] = logr_min + (logr_max-logr_min)/(NGRAV1DPROF-1)*i;
    menc[i] = Interpolate1DArrayEven(rad,menc_init, pow(10,logr[i]), NGRAV1DPROF );
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
    std::cout << "GMenv="<< GMenv << "\n";
    std::cout << "rstar_initial = "<< rstar_initial<<"\n";
    std::cout << "mstar_initial = "<< mstar_initial<<"\n";
    std::cout << "t_relax ="<<t_relax<<"\n";
  }
  

} // end






// Source Function for two point masses
void MyPointMass(MeshBlock *pmb, const Real time, const Real dt,  const AthenaArray<Real> *flux,
		  const AthenaArray<Real> &prim,
		  const AthenaArray<Real> &prim_scalar, const AthenaArray<Real> &bcc,
		  AthenaArray<Real> &cons, AthenaArray<Real> &cons_scalar) 
{ 

  // Gravitational acceleration 
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
	  
	//
	//  COMPUTE ACCELERATIONS 
	//
	// PM1
	Real GMenc1 = Ggrav*Interpolate1DArrayEven(logr,menc,log10(r) , NGRAV1DPROF);
	Real a_r1 = -GMenc1*pmb->pcoord->coord_src1_i_(i)/r;

	// add the PM1 accel
	Real a_r  = a_r1;
	Real a_th = 0.0;
	Real a_ph = 0.0;
	
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




//======================================================================================
//! \fn void MeshBlock::ProblemGenerator(ParameterInput *pin)
//  \brief Spherical Coords HSE Envelope problem generator
//======================================================================================

void MeshBlock::ProblemGenerator(ParameterInput *pin)
{

  // local vars
  Real den, pres;

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
	Real cos_th = cos(th);
	Real Rcyl = r*sin_th;
	
	// get the density
	den = Interpolate1DArrayEven(rad,rho, r , NARRAY);
	den = std::max(den,da);
	
	// get the pressure 
	pres = Interpolate1DArrayEven(rad,p, r , NARRAY);
	pres = std::max(pres,pa);

	// set the density
	phydro->u(IDN,k,j,i) = den;
	
   	// set the momenta components
	phydro->u(IM1,k,j,i) = 0.0;
	phydro->u(IM2,k,j,i) = 0.0;
	phydro->u(IM3,k,j,i) = 0.0;
	
	//set the energy 
	phydro->u(IEN,k,j,i) = pres/(gamma_gas-1);
	phydro->u(IEN,k,j,i) += 0.5*(SQR(phydro->u(IM1,k,j,i))+SQR(phydro->u(IM2,k,j,i))
				     + SQR(phydro->u(IM3,k,j,i)))/phydro->u(IDN,k,j,i);

       
      }
    }
  } // end loop over cells
  return;
} // end ProblemGenerator

//======================================================================================
//! \fn void MeshBlock::UserWorkInLoop(void)
//  \brief Function called once every time step for user-defined work.
//======================================================================================
void MeshBlock::UserWorkInLoop(void)
{
  Real time = pmy_mesh->time;
  Real dt = pmy_mesh->dt;
  Real tau;

  // Add timestep diagnostics
  if(pmy_mesh->ncycle % 10 == 0){
    if(new_block_dt_ == pmy_mesh->dt){
      // call NewBlockTimeStep with extra diagnostic output
      phydro->NewBlockTimeStep(1);
    }
  }  


  
  // if less than the relaxation time, apply 
  // a damping to the fluid velocities
  if(time < t_relax){
    tau = tau_relax_start;
    Real dex = log10(tau_relax_end)-log10(tau_relax_start);
    if(time > 0.2*t_relax){
      tau *= pow(10, dex*(time-0.2*t_relax)/(0.8*t_relax) );
    }
    if (Globals::my_rank==0){
      std::cout << "Relaxing: tau_damp ="<<tau<<std::endl;
    }
  } // time<t_relax

  for (int k=ks; k<=ke; k++) {
    Real ph= pcoord->x3v(k);
    for (int j=js; j<=je; j++) {
      Real th= pcoord->x2v(j);
      for (int i=is; i<=ie; i++) {
	Real r = pcoord->x1v(i);
	Real den = phydro->u(IDN,k,j,i);
	Real GMenc1 = Ggrav*Interpolate1DArrayEven(logr,menc,log10(r) , NGRAV1DPROF);	

	if (time<t_relax){
	  Real vr  = phydro->u(IM1,k,j,i) / den;
	  Real vth = phydro->u(IM2,k,j,i) / den;
	  Real vph = phydro->u(IM3,k,j,i) / den;
	  Real a_damp_r =  - vr/tau;
	  Real a_damp_th = - vth/tau;
	  Real a_damp_ph = - vph/tau;

	  phydro->u(IM1,k,j,i) += dt*den*a_damp_r;
	  phydro->u(IM2,k,j,i) += dt*den*a_damp_th;
	  phydro->u(IM3,k,j,i) += dt*den*a_damp_ph;
	  
	  phydro->u(IEN,k,j,i) += dt*den*a_damp_r*vr + dt*den*a_damp_th*vth + dt*den*a_damp_ph*vph; 
		
	}//end time<t_relax
	
      }
    }
  } // end loop over cells                   
  

  return;
} // end of UserWorkInLoop


//========================================================================================
// MM
//! \fn void MeshBlock::MeshUserWorkInLoop(void)
//  \brief Function called once every time step for user-defined work.
//========================================================================================

void Mesh::MeshUserWorkInLoop(ParameterInput *pin){
  Mesh *pm = my_blocks(0)->pmy_mesh;
  // sum the enclosed mass profile for monopole gravity
  if(ncycle%update_grav_every == 0){
    SumMencProfile(pm,menc);
    if (Globals::my_rank == 0 ){
      std::cout << "enclosed mass updated... Menc(r=rstar_initial) = " << Interpolate1DArrayEven(logr,menc,log10(rstar_initial), NGRAV1DPROF) <<"\n";
    }
  }
}


void SumMencProfile(Mesh *pm, Real (&menc)[NGRAV1DPROF]){

  Real m1 =  GM1/Ggrav;
  // start by setting enclosed mass at each radius to zero
  for (int ii = 0; ii <NGRAV1DPROF; ii++){
    menc[ii] = 0.0;
  }
  
  MeshBlock *pmb=pm->my_blocks(0);
  AthenaArray<Real> vol;
  
  int ncells1 = pmb->block_size.nx1 + 2*(NGHOST);
  vol.NewAthenaArray(ncells1);

  // Loop over MeshBlocks
  for (int b=0; b<pm->nblocal; ++b) {
    pmb = pm->my_blocks(b);
    Hydro *phyd = pmb->phydro;

    // Sum history variables over cells.  Note ghost cells are never included in sums
    for (int k=pmb->ks; k<=pmb->ke; ++k) {
      for (int j=pmb->js; j<=pmb->je; ++j) {
	pmb->pcoord->CellVolume(k,j,pmb->is,pmb->ie,vol);
	for (int i=pmb->is; i<=pmb->ie; ++i) {
	  // cell mass dm
	  Real dm = vol(i) * phyd->u(IDN,k,j,i);
	  Real logr_cell = log10( pmb->pcoord->x1v(i) );

	  // loop over radii in profile
	  for (int ii = 0; ii <NGRAV1DPROF; ii++){
	    if( logr_cell < logr[ii] ){
	      menc[ii] += dm;
	    }
	  }
	  
	}
      }
    }//end loop over cells
  }//end loop over meshblocks

#ifdef MPI_PARALLEL
  // sum over all ranks, add m1
  if (Globals::my_rank == 0) {
    for (int ii = 0; ii <NGRAV1DPROF; ii++){
      menc[ii] += m1;
    }
    MPI_Reduce(MPI_IN_PLACE, menc, NGRAV1DPROF, MPI_ATHENA_REAL, MPI_SUM, 0,MPI_COMM_WORLD);
  } else {
    MPI_Reduce(menc,menc,NGRAV1DPROF, MPI_ATHENA_REAL, MPI_SUM, 0,MPI_COMM_WORLD);
  }

  
  // and broadcast the result
  MPI_Bcast(menc,NGRAV1DPROF,MPI_ATHENA_REAL,0,MPI_COMM_WORLD);
#endif
    
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

  return;
}






// 1D Interpolation that assumes EVEN spacing in x array

Real Interpolate1DArrayEven(Real *x,Real *y,Real x0, int length){ 
  // check the lower bound
  if(x[0] >= x0){
    //std::cout << "hit lower bound!\n";
    return y[0];
  }
  // check the upper bound
  if(x[length-1] <= x0){
    //std::cout << "hit upper bound!\n";
    return y[length-1];
  }

  int i = floor( (x0-x[0])/(x[1]-x[0]) );
  
  // if in the interior, do a linear interpolation
  if (x[i+1] >= x0){ 
    Real dx =  (x[i+1]-x[i]);
    Real d = (x0 - x[i]);
    Real s = (y[i+1]-y[i]) /dx;
    return s*d + y[i];
  }
  // should never get here, -9999.9 represents an error
  return -9999.9;
}


