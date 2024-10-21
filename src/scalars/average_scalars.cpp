//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file average_conserved.cpp
//  \brief Implements polar boundary averaging by Morgan MacLeod.

// Athena++ headers
#include "scalars.hpp"
#include "../hydro/hydro.hpp"
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../coordinates/coordinates.hpp"
#include "../field/field.hpp"
#include "../mesh/mesh.hpp"
#include "../bvals/bvals.hpp"
#include "../reconstruct/reconstruction.hpp"
#include <stdexcept>

// OpenMP header
#ifdef OPENMP_PARALLEL
#include <omp.h>
#include <math.h>
#endif



//----------------------------------------------------------------------------------------
//! \fn  void PassiveScalars::PhiAverageConserved
//  \brief averages conserved quantities in the phi direction
// should match up with Hydro::PhiAverageConserved

void PassiveScalars::PhiAverageScalars(AthenaArray<Real> &s_in,AthenaArray<Real> &s_out)
{
  // check NSCALARS > 0
  if(NSCALARS == 0) return;
  // check if we want polar average
  if(do_average_ == false) return;
  // check if we're on the x2-boundary
  MeshBlock *pmb=pmy_block;
  if((pmb->pbval->block_bcs[BoundaryFace::inner_x2] == BoundaryFlag::block) &&
     (pmb->pbval->block_bcs[BoundaryFace::outer_x2] == BoundaryFlag::block)) {
    return;
  }

  Get_block_N_zone_avg_Scalars(pmb);
      
  int is = pmb->is; int js = pmb->js; int ks = pmb->ks;
  int ie = pmb->ie; int je = pmb->je; int ke = pmb->ke;
  Real dphi = pmb->pcoord->dx3f(ks); // assume that cell size in phi is constant

  for (int n=0; n<NSCALARS; ++n) {
    for (int j=js; j<=je; ++j) {
      int n_avg_loops = pmb->block_size.nx3/n_avg_(j);
      for (int l=1;l<=n_avg_loops;++l){
	int ks_avg = ks + (l-1)*n_avg_(j);
	int ke_avg = ks_avg + n_avg_(j) -1;
	for (int i=is; i<=ie; ++i) {
	  Real s_k_avg = 0.0;
	  for (int k=ks_avg; k<=ke_avg; ++k) {
	    s_k_avg +=  s_in(n,k,j,i);
	  }
	  // set the new value
	  for (int k=ks_avg; k<=ke_avg; ++k) {
	    s_out(n,k,j,i) = s_k_avg/n_avg_(j);
	  }
	}
      }
    }
  }

  return;
}


// NOTE: This should be de-duplicated with hydro version
void PassiveScalars::Get_block_N_zone_avg_Scalars(MeshBlock *pmb){

  int is = pmb->is; int js = pmb->js; int ks = pmb->ks;
  int ie = pmb->ie; int je = pmb->je; int ke = pmb->ke;

  for (int j=js; j<=je; ++j) {
    n_avg_(j) = 1;
  }
  
  
  //check if we're on the inner x2-boundary
  if(pmb->pbval->block_bcs[BoundaryFace::inner_x2] == BoundaryFlag::polar) {
    for (int j=js; j<=je; ++j) {
      //int n_avg_temp = pmb->block_size.nx3/pow(2,(j-js));
      int n_avg_temp = round(pmb->block_size.nx3/pow(2,round(log2(j-js+1))));
      if( n_avg_temp>1){     
	n_avg_(j) = n_avg_temp;
      }else{
	n_avg_(j) = 1;
      }
    }
  }
  if (pmb->pbval->block_bcs[BoundaryFace::outer_x2] == BoundaryFlag::polar)  {
    // average the je index
    for (int j=js; j<=je; ++j) {
      //int n_avg_temp = pmb->block_size.nx3/pow(2,(je-j));
      int n_avg_temp = round(pmb->block_size.nx3/pow(2,round(log2(je-j+1))));
      if( n_avg_temp>1){     
        n_avg_(j) = n_avg_temp;
      }else{
	n_avg_(j) = 1;
      }
    }    
  }

  
}


