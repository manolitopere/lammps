/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */
/* ------------------------------------------------------------------------
   Contributing authors: Thomas Swinburne (CNRS & CINaM, Marseille, France)

   Please cite the related publication:
   T.D. Swinburne and M.-C. Marinica, Unsupervised calculation of free energy barriers in large crystalline systems, Physical Review Letters 2018
------------------------------------------------------------------------- */


#include "fix_pafi_ti.h"
#include <mpi.h>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include "math_extra.h"
#include "atom.h"
#include "force.h"
#include "pair_hybrid.h"
#include "update.h"
#include "modify.h"
#include "domain.h"
#include "respa.h"
#include "comm.h"
#include "compute.h"
#include "random_mars.h"
#include "memory.h"
#include "error.h"
#include "utils.h"
#include "citeme.h"

using namespace LAMMPS_NS;

static const char cite_fix_pafi_package[] =
  "citation for fix pafiTI:\n\n"
  "@article{SwinburneMarinica2018,\n"
  "author={T. D. Swinburne and M. C. Marinica},\n"
  "title={Unsupervised calculation of free energy barriers in large "
  "crystalline systems},\n"
  "journal={Physical Review Letters},\n"
  "volume={120},\n"
  "number={13},\n"
  "pages={135503},\n"
  "year={2018},\n"
  "publisher={APS}\n}\n"
  "Recommended to be coupled with PAFI++ code:\n"
  "https://github.com/tomswinburne/pafiTI\n";

using namespace FixConst;

/* ---------------------------------------------------------------------- */

FixPAFITI::FixPAFITI(LAMMPS *lmp, int narg, char **arg) :
  Fix(lmp, narg, arg), random(NULL), computename(NULL),
      h(NULL), step_respa(NULL)
{
  if (lmp->citeme) lmp->citeme->add(cite_fix_pafi_package);

  if (narg < 11) error->all(FLERR,"To few arguments for fix pafiTI command");

  dynamic_group_allow = 0;
  vector_flag = 1;
  size_vector = 6;
  global_freq = 1;
  extvector = 0;
  od_flag = 0;
  com_flag = 1;
  time_integrate = 1;

  int n = strlen(arg[3])+1;
  computename = new char[n];
  strcpy(computename,&arg[3][0]);

  icompute = modify->find_compute(computename);
  if (icompute == -1)
    error->all(FLERR,"Compute ID for fix pafiTI does not exist");
  PathCompute = modify->compute[icompute];
  if (PathCompute->peratom_flag==0)
    error->all(FLERR,"Compute for fix pafiTI does not calculate a local array");
  if (PathCompute->size_peratom_cols < 9)
    error->all(FLERR,"Compute for fix pafiTI must have 9 fields per atom");

  if (comm->me==0) {
    if (screen) fprintf(screen,
      "fix pafiTI compute name,style: %s,%s\n",computename,PathCompute->style);
    if (logfile) fprintf(logfile,
      "fix pafiTI compute name,style: %s,%s\n",computename,PathCompute->style);
  }

  PairHybrid *hybrid = NULL;
  if (strncmp(force->pair_style,"hybrid",6) == 0) {
    hybrid = (PairHybrid *)force->pair;
  } else error->all(FLERR,"This compute will only work with a hybrid pair style");


  respa_level_support = 1;
  ilevel_respa = nlevels_respa = 0;

  temperature = force->numeric(FLERR,arg[4]);
  t_period = force->numeric(FLERR,arg[5]);
  seed = force->inumeric(FLERR,arg[6]);

  int ntypes = atom->ntypes;
  int nstyles = hybrid->nstyles;

  memory->create(scale,nstyles,ntypes+1,ntypes+1,"FixPAFITI:scale");

  ref_pair = force->inumeric(FLERR,arg[7])-1;
  if(ref_pair>=hybrid->nstyles) error->all(FLERR,"Reference pair index not found");

  lambda = force->numeric(FLERR,arg[8]);
  if(lambda <0.0 || lambda>1.0) error->all(FLERR,"Invalid lambda value");

  for (int m = 0; m < nstyles; m++) {
    int pdim;
    void *ptr = hybrid->styles[m]->extract("scale",pdim);
    scale[m] = (double **) ptr;
    for(int ii=0;ii<ntypes+1;ii++) for(int jj=0;jj<ntypes+1;jj++) {
      if(m==ref_pair) scale[m][ii][jj] = 1.0-lambda;
      else scale[m][ii][jj] = lambda;
    }
  }
  // TODO UNITS
  gamma = 1. / t_period / force->ftm2v;
  sqrtD = sqrt(1.) * sqrt(24.0*force->boltz/t_period/update->dt/force->mvv2e*temperature) / force->ftm2v;

  // optional args
  int iarg = 9;
  while (iarg < narg) {
    if (strcmp(arg[iarg],"overdamped") == 0) {
      if (strcmp(arg[iarg+1],"no") == 0) od_flag = 0;
      else if (strcmp(arg[iarg+1],"0") == 0) od_flag = 0;
      else if (strcmp(arg[iarg+1],"yes") == 0) od_flag = 1;
      else if (strcmp(arg[iarg+1],"1") == 0) od_flag = 1;
      iarg += 2;
    } else if (strcmp(arg[iarg],"com") == 0) {
      if (strcmp(arg[iarg+1],"no") == 0) com_flag = 0;
      else if (strcmp(arg[iarg+1],"0") == 0) com_flag = 0;
      else if (strcmp(arg[iarg+1],"yes") == 0) com_flag = 1;
      else if (strcmp(arg[iarg+1],"1") == 0) com_flag = 1;
      iarg += 2;
    } else error->all(FLERR,"Illegal fix pafiTI command");
  }
  force_flag = 0;

  for(int i = 0; i < 10; i++) {
    c_v[i] = 0.;
    c_v_all[i] = 0.;
  }
  for(int i=0; i<6; i++) {
    proj[i] = 0.0;
    proj_all[i] = 0.0;
  }
  for(int i=0; i<5; i++) {
    results[i] = 0.0;
    results_all[i] = 0.0;
  }
  maxatom = 1;
  memory->create(h,maxatom,3,"FixPAFITI:h");

  // initialize Marsaglia RNG with processor-unique seed
  random = new RanMars(lmp,seed + comm->me);

}

/* ---------------------------------------------------------------------- */

FixPAFITI::~FixPAFITI()
{
  if (copymode) return;
  delete random;
  delete [] computename;
  memory->destroy(h);
}

/* ---------------------------------------------------------------------- */

int FixPAFITI::setmask()
{
  int mask = 0;
  //
  mask |= PRE_FORCE;
  mask |= PRE_FORCE_RESPA;
  // projection
  mask |= POST_FORCE;
  mask |= POST_FORCE_RESPA;
  mask |= MIN_POST_FORCE;
  // nve
  mask |= INITIAL_INTEGRATE;
  mask |= FINAL_INTEGRATE;
  mask |= INITIAL_INTEGRATE_RESPA;
  mask |= FINAL_INTEGRATE_RESPA;
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixPAFITI::init()
{
  dtv = update->dt;
  dtf = 0.5 * update->dt * force->ftm2v;

  icompute = modify->find_compute(computename);
  if (icompute == -1)
    error->all(FLERR,"Compute ID for fix pafiTI does not exist");
  PathCompute = modify->compute[icompute];
  if (PathCompute->peratom_flag==0)
    error->all(FLERR,"Compute for fix pafiTI does not calculate a local array");
  if (PathCompute->size_peratom_cols < 9)
    error->all(FLERR,"Compute for fix pafiTI must have 9 fields per atom");


  if (strstr(update->integrate_style,"respa")) {
    step_respa = ((Respa *) update->integrate)->step; // nve
    nlevels_respa = ((Respa *) update->integrate)->nlevels;
    if (respa_level >= 0) ilevel_respa = MIN(respa_level,nlevels_respa-1);
    else ilevel_respa = nlevels_respa-1;
  }

}

void FixPAFITI::setup(int vflag)
{
  if (strstr(update->integrate_style,"verlet"))
    post_force(vflag);
  else
    for (int ilevel = 0; ilevel < nlevels_respa; ilevel++) {
      ((Respa *) update->integrate)->copy_flevel_f(ilevel);
      post_force_respa(vflag,ilevel,0);
      ((Respa *) update->integrate)->copy_f_flevel(ilevel);
    }
}

void FixPAFITI::min_setup(int vflag)
{
  if( utils::strmatch(update->minimize_style,"^fire")==0 &&
          utils::strmatch(update->minimize_style,"^quickmin")==0 )
    error->all(FLERR,"fix pafiTI requires a damped dynamics minimizer");
  min_post_force(vflag);
}


void FixPAFITI::pre_force(int vflag)
{
  double **f = atom->f;
  double *rmass = atom->rmass;
  double *mass = atom->mass;
  int *type = atom->type;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;

  for (int i = 0; i < nlocal; i++) {
    if (mask[i] & groupbit) {
      f[i][0] = 0.0;
      f[i][1] = 0.0;
      f[i][2] = 0.0;
    }
  }
  PairHybrid *hybrid = NULL;
  if (strncmp(force->pair_style,"hybrid",6) == 0) {
    hybrid = (PairHybrid *)force->pair;
  } else error->all(FLERR,"This compute will only work with a hybrid pair style");

  int nstyles = hybrid->nstyles;

  hybrid->styles[ref_pair]->compute(1,1);
  for (int i = 0; i < nlocal; i++) {
    if (mask[i] & groupbit) {
      f[i][0] *= -lambda/(1.0-lambda);
      f[i][1] *= -lambda/(1.0-lambda);
      f[i][2] *= -lambda/(1.0-lambda);
    }
  }

  for (int m = 0; m < nstyles; m++) if(m!=ref_pair) hybrid->styles[m]->compute(1,1);
  for (int i = 0; i < nlocal; i++) {
    if (mask[i] & groupbit) {
      f[i][0] *= 1.0/lambda;
      f[i][1] *= 1.0/lambda;
      f[i][2] *= 1.0/lambda;
    }
  }
  /* record */

  for (int i = 0; i < nlocal; i++) {
    if (mask[i] & groupbit) {
      f[i][0] = 0.0;
      f[i][1] = 0.0;
      f[i][2] = 0.0;
    }
  }

};


void FixPAFITI::post_force(int vflag)
{
  double **x = atom->x;
  double **v = atom->v;
  double **f = atom->f;
  double *rmass = atom->rmass;
  double *mass = atom->mass;
  int *type = atom->type;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;

  // reallocate norm array if necessary
  if (atom->nmax > maxatom) {
    maxatom = atom->nmax;
    memory->destroy(h);
    memory->create(h,maxatom,3,"FixPAFITI:h");
  }

  PathCompute->compute_peratom();
  double **path = PathCompute->array_atom;

  double xum=0.;

  // proj 0,1,2 = f.n, v.n, h.n
  // proj 3,4,5 = psi, f.n**2, f*(1-psi)
  // c_v 0,1,2 = fxcom, fycom, fzcom etc
  for(int i = 0; i < 10; i++) {
    c_v[i] = 0.;
    c_v_all[i] = 0.;
  }
  for(int i = 0; i < 6; i++) {
    proj[i] = 0.;
    proj_all[i] = 0.;
  }

  double deviation[3] = {0.,0.,0.};

  double fn;

  force_flag=0;
  for (int i = 0; i < nlocal; i++) {
    if (mask[i] & groupbit) {
      h[i][0] = random->uniform() - 0.5;
      h[i][1] = random->uniform() - 0.5;
      h[i][2] = random->uniform() - 0.5;

      proj[0] += f[i][0] * path[i][3]; // f.n
      proj[0] += f[i][1] * path[i][4]; // f.n
      proj[0] += f[i][2] * path[i][5]; // f.n

      proj[1] += v[i][0] * path[i][3]; // v.n
      proj[1] += v[i][1] * path[i][4]; // v.n
      proj[1] += v[i][2] * path[i][5]; // v.n

      proj[2] += h[i][0] * path[i][3]; // h.n
      proj[2] += h[i][1] * path[i][4]; // h.n
      proj[2] += h[i][2] * path[i][5]; // h.n

      deviation[0] = x[i][0]-path[i][0]; // x-path
      deviation[1] = x[i][1]-path[i][1]; // x-path
      deviation[2] = x[i][2]-path[i][2]; // x-path
      domain->minimum_image(deviation);

      proj[3] += path[i][6]*deviation[0]; // (x-path).dn/nn = psi
      proj[3] += path[i][7]*deviation[1]; // (x-path).dn/nn = psi
      proj[3] += path[i][8]*deviation[2]; // (x-path).dn/nn = psi

      proj[4] += path[i][3]*deviation[0]; // (x-path).n
      proj[4] += path[i][4]*deviation[1]; // (x-path).n
      proj[4] += path[i][5]*deviation[2]; // (x-path).n

      proj[5] += f[i][3]*deviation[0]; // (x-path).f
      proj[5] += f[i][4]*deviation[1]; // (x-path).f
      proj[5] += f[i][5]*deviation[2]; // (x-path).f

    }
  }

  if(com_flag == 0){
    c_v[9] += 1.0;
  } else {
    for (int i = 0; i < nlocal; i++)
      if (mask[i] & groupbit) {

        c_v[0] += f[i][0];
        c_v[1] += f[i][1];
        c_v[2] += f[i][2];

        c_v[3] += v[i][0];
        c_v[4] += v[i][1];
        c_v[5] += v[i][2];

        c_v[6] += h[i][0];
        c_v[7] += h[i][1];
        c_v[8] += h[i][2];

        c_v[9] += 1.0;
      }
  }
  MPI_Allreduce(proj,proj_all,6,MPI_DOUBLE,MPI_SUM,world);
  MPI_Allreduce(c_v,c_v_all,10,MPI_DOUBLE,MPI_SUM,world);

  // results - f.n*(1-psi), (f.n)^2*(1-psi)^2, 1-psi, dX.n
  results_all[0] = proj_all[0] * (1.-proj_all[3]);
  results_all[1] = results_all[0] * results_all[0];
  results_all[2] = 1.-proj_all[3];
  results_all[3] = fabs(proj_all[4]);
  results_all[4] = proj_all[5]; // dX.f
  force_flag = 1;

  for (int i = 0; i < nlocal; i++){
    if (mask[i] & groupbit) {

      f[i][0] -= proj_all[0] * path[i][3] + c_v_all[0]/c_v_all[9];
      f[i][1] -= proj_all[0] * path[i][4] + c_v_all[1]/c_v_all[9];
      f[i][2] -= proj_all[0] * path[i][5] + c_v_all[2]/c_v_all[9];

      v[i][0] -= proj_all[1] * path[i][3] + c_v_all[3]/c_v_all[9];
      v[i][1] -= proj_all[1] * path[i][4] + c_v_all[4]/c_v_all[9];
      v[i][2] -= proj_all[1] * path[i][5] + c_v_all[5]/c_v_all[9];

      h[i][0] -= proj_all[2] * path[i][3] + c_v_all[6]/c_v_all[9];
      h[i][1] -= proj_all[2] * path[i][4] + c_v_all[7]/c_v_all[9];
      h[i][2] -= proj_all[2] * path[i][5] + c_v_all[8]/c_v_all[9];
    }
  }

  if (od_flag == 0) {
    for (int i = 0; i < nlocal; i++){
      if (mask[i] & groupbit) {
        if(rmass) mass_f = sqrt(rmass[i]);
        else mass_f = sqrt(mass[type[i]]);

        f[i][0] += -gamma * mass_f * mass_f * v[i][0];
        f[i][1] += -gamma * mass_f * mass_f * v[i][1];
        f[i][2] += -gamma * mass_f * mass_f * v[i][2];

        f[i][0] += sqrtD * mass_f * h[i][0];
        f[i][1] += sqrtD * mass_f * h[i][1];
        f[i][2] += sqrtD * mass_f * h[i][2];
      }
    }
  } else {
    for (int i = 0; i < nlocal; i++){
      if (mask[i] & groupbit) {

        if(rmass) mass_f = sqrt(rmass[i]);
        else mass_f = sqrt(mass[type[i]]);

        f[i][0] += sqrtD * h[i][0] * mass_f;
        f[i][1] += sqrtD * h[i][1] * mass_f;
        f[i][2] += sqrtD * h[i][2] * mass_f;

        f[i][0] /=  gamma * mass_f * mass_f;
        f[i][1] /=  gamma * mass_f * mass_f;
        f[i][2] /=  gamma * mass_f * mass_f;

      }
    }
  }

};

void FixPAFITI::post_force_respa(int vflag, int ilevel, int iloop)
{
  // set force to desired value on requested level, 0.0 on other levels

  if (ilevel == ilevel_respa) post_force(vflag);
  else {
    double **x = atom->x;
    double **f = atom->f;
    int *mask = atom->mask;
    int nlocal = atom->nlocal;

    for (int i = 0; i < nlocal; i++)
      if (mask[i] & groupbit) {
        for (int k = 0; k < 3; k++) f[i][k] = 0.0;
      }
  }
};

void FixPAFITI::min_post_force(int vflag)
{
  double **x = atom->x;
  double **v = atom->v;
  double **f = atom->f;
  double *rmass = atom->rmass;
  double *mass = atom->mass;
  int *type = atom->type;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;

  PathCompute->compute_peratom();
  double **path = PathCompute->array_atom;

  double xum=0.;

  // proj 0,1,2 = f.n, v.n, h.n
  // proj 3,4,5 = psi, f.n**2, f*(1-psi)
  // c_v 0,1,2 = fxcom, fycom, fzcom etc
  for(int i = 0; i < 10; i++) {
    c_v[i] = 0.;
    c_v_all[i] = 0.;
  }
  for(int i = 0; i < 6; i++) {
    proj[i] = 0.;
    proj_all[i] = 0.;
  }

  double deviation[3] = {0.,0.,0.};

  double fn;

  force_flag=0;
  for (int i = 0; i < nlocal; i++) {
    if (mask[i] & groupbit) {

      proj[0] += f[i][0] * path[i][3]; // f.n
      proj[0] += f[i][1] * path[i][4]; // f.n
      proj[0] += f[i][2] * path[i][5]; // f.n

      proj[1] += v[i][0] * path[i][3]; // v.n
      proj[1] += v[i][1] * path[i][4]; // v.n
      proj[1] += v[i][2] * path[i][5]; // v.n

      proj[2] += h[i][0] * path[i][3]; // h.n
      proj[2] += h[i][1] * path[i][4]; // h.n
      proj[2] += h[i][2] * path[i][5]; // h.n

      deviation[0] = x[i][0]-path[i][0]; // x-path
      deviation[1] = x[i][1]-path[i][1]; // x-path
      deviation[2] = x[i][2]-path[i][2]; // x-path
      domain->minimum_image(deviation);

      proj[3] += path[i][6]*deviation[0]; // (x-path).dn/nn = psi
      proj[3] += path[i][7]*deviation[1]; // (x-path).dn/nn = psi
      proj[3] += path[i][8]*deviation[2]; // (x-path).dn/nn = psi

      proj[4] += path[i][3]*deviation[0]; // (x-path).n
      proj[4] += path[i][4]*deviation[1]; // (x-path).n
      proj[4] += path[i][5]*deviation[2]; // (x-path).n

      proj[5] += f[i][3]*deviation[0]; // (x-path).f
      proj[5] += f[i][4]*deviation[1]; // (x-path).f
      proj[5] += f[i][5]*deviation[2]; // (x-path).f

    }
  }

  if(com_flag == 0){
    c_v[9] += 1.0;
  } else {
    for (int i = 0; i < nlocal; i++)
      if (mask[i] & groupbit) {

        c_v[0] += f[i][0];
        c_v[1] += f[i][1];
        c_v[2] += f[i][2];

        c_v[3] += v[i][0];
        c_v[4] += v[i][1];
        c_v[5] += v[i][2];

        c_v[6] += h[i][0];
        c_v[7] += h[i][1];
        c_v[8] += h[i][2];

        c_v[9] += 1.0;
      }
  }
  MPI_Allreduce(proj,proj_all,6,MPI_DOUBLE,MPI_SUM,world);
  MPI_Allreduce(c_v,c_v_all,10,MPI_DOUBLE,MPI_SUM,world);

  results_all[0] = proj_all[0] * (1.-proj_all[3]); // f.n * psi
  results_all[1] = results_all[0] * results_all[0]; // (f.n * psi)^2
  results_all[2] = 1.-proj_all[3]; // psi
  results_all[3] = fabs(proj_all[4]); // dX.n
  results_all[4] = proj_all[5]; // dX.f

  MPI_Bcast(results_all,5,MPI_DOUBLE,0,world);
  force_flag = 1;

  for (int i = 0; i < nlocal; i++){
    if (mask[i] & groupbit) {

      f[i][0] -= proj_all[0] * path[i][3] + c_v_all[0]/c_v_all[9];
      f[i][1] -= proj_all[0] * path[i][4] + c_v_all[1]/c_v_all[9];
      f[i][2] -= proj_all[0] * path[i][5] + c_v_all[2]/c_v_all[9];

      v[i][0] -= proj_all[1] * path[i][3] + c_v_all[3]/c_v_all[9];
      v[i][1] -= proj_all[1] * path[i][4] + c_v_all[4]/c_v_all[9];
      v[i][2] -= proj_all[1] * path[i][5] + c_v_all[5]/c_v_all[9];

    }
  }
};

double FixPAFITI::compute_vector(int n)
{
  return results_all[n];
};

void FixPAFITI::initial_integrate(int vflag)
{
  double dtfm;

  // update v and x of atoms in group

  double **x = atom->x;
  double **v = atom->v;
  double **f = atom->f;

  double *rmass = atom->rmass;
  double *mass = atom->mass;
  int *type = atom->type;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;
  if (igroup == atom->firstgroup) nlocal = atom->nfirst;

  PathCompute->compute_peratom();
  double **path = PathCompute->array_atom;

  for(int i = 0; i < 10; i++) {
    c_v[i] = 0.;
    c_v_all[i] = 0.;
  }
  for(int i = 0; i < 6; i++) {
    proj[i] = 0.;
    proj_all[i] = 0.;
  }

  for (int i = 0; i < nlocal; i++) {
    if (mask[i] & groupbit) {
      proj[0] += f[i][0] * path[i][3]; // f.n
      proj[0] += f[i][1] * path[i][4]; // f.n
      proj[0] += f[i][2] * path[i][5]; // f.n

      proj[1] += v[i][0] * path[i][3]; // v.n
      proj[1] += v[i][1] * path[i][4]; // v.n
      proj[1] += v[i][2] * path[i][5]; // v.n
    }
  }
  if(com_flag == 0){
    c_v[9] += 1.0;
  } else {
    for (int i = 0; i < nlocal; i++)
      if (mask[i] & groupbit) {

        c_v[0] += v[i][0];
        c_v[1] += v[i][1];
        c_v[2] += v[i][2];

        c_v[3] += f[i][0];
        c_v[4] += f[i][1];
        c_v[5] += f[i][2];

        c_v[9] += 1.0;
      }
  }


  MPI_Allreduce(proj,proj_all,5,MPI_DOUBLE,MPI_SUM,world);
  MPI_Allreduce(c_v,c_v_all,10,MPI_DOUBLE,MPI_SUM,world);

  if (od_flag == 0){
    if (rmass) {
      for (int i = 0; i < nlocal; i++)
        if (mask[i] & groupbit) {
          dtfm = dtf / rmass[i];
          v[i][0] += dtfm * (f[i][0]-path[i][3]*proj_all[0] - c_v_all[3]/c_v_all[9]);
          v[i][1] += dtfm * (f[i][1]-path[i][4]*proj_all[0] - c_v_all[4]/c_v_all[9]);
          v[i][2] += dtfm * (f[i][2]-path[i][5]*proj_all[0] - c_v_all[5]/c_v_all[9]);
          x[i][0] += dtv * (v[i][0]-path[i][3]*proj_all[1] - c_v_all[0]/c_v_all[9]);
          x[i][1] += dtv * (v[i][1]-path[i][4]*proj_all[1] - c_v_all[1]/c_v_all[9]);
          x[i][2] += dtv * (v[i][2]-path[i][5]*proj_all[1] - c_v_all[2]/c_v_all[9]);
        }
    } else {
      for (int i = 0; i < nlocal; i++)
        if (mask[i] & groupbit) {
          dtfm = dtf / mass[type[i]];
          v[i][0] += dtfm * (f[i][0]-path[i][3]*proj_all[0] - c_v_all[3]/c_v_all[9]);
          v[i][1] += dtfm * (f[i][1]-path[i][4]*proj_all[0] - c_v_all[4]/c_v_all[9]);
          v[i][2] += dtfm * (f[i][2]-path[i][5]*proj_all[0] - c_v_all[5]/c_v_all[9]);
          x[i][0] += dtv * (v[i][0]-path[i][3]*proj_all[1] - c_v_all[0]/c_v_all[9]);
          x[i][1] += dtv * (v[i][1]-path[i][4]*proj_all[1] - c_v_all[1]/c_v_all[9]);
          x[i][2] += dtv * (v[i][2]-path[i][5]*proj_all[1] - c_v_all[2]/c_v_all[9]);
        }
    }
  } else {
    if (rmass) {
      for (int i = 0; i < nlocal; i++)
        if (mask[i] & groupbit) {
          dtfm = dtf / rmass[i];
          v[i][0] = 0.;
          v[i][1] = 0.;
          v[i][2] = 0.;
          x[i][0] += dtv * (f[i][0]-path[i][3]*proj_all[0] - c_v_all[3]/c_v_all[9]);
          x[i][1] += dtv * (f[i][1]-path[i][4]*proj_all[0] - c_v_all[4]/c_v_all[9]);
          x[i][2] += dtv * (f[i][2]-path[i][5]*proj_all[0] - c_v_all[5]/c_v_all[9]);
        }
    } else {
      for (int i = 0; i < nlocal; i++)
        if (mask[i] & groupbit) {
          dtfm = dtf / mass[type[i]];
          v[i][0] = 0.;
          v[i][1] = 0.;
          v[i][2] = 0.;
          x[i][0] += dtv * (f[i][0]-path[i][3]*proj_all[0] - c_v_all[3]/c_v_all[9]);
          x[i][1] += dtv * (f[i][1]-path[i][4]*proj_all[0] - c_v_all[4]/c_v_all[9]);
          x[i][2] += dtv * (f[i][2]-path[i][5]*proj_all[0] - c_v_all[5]/c_v_all[9]);
        }
    }
  }
};

/* ---------------------------------------------------------------------- */

void FixPAFITI::final_integrate()
{
  double dtfm;

  // update v of atoms in group
  double **v = atom->v;
  double **f = atom->f;
  double *rmass = atom->rmass;
  double *mass = atom->mass;
  int *type = atom->type;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;
  if (igroup == atom->firstgroup) nlocal = atom->nfirst;

  PathCompute->compute_peratom();
  double **path = PathCompute->array_atom;

  for(int i = 0; i < 10; i++) {
    c_v[i] = 0.;
    c_v_all[i] = 0.;
  }
  for(int i = 0; i < 6; i++) {
    proj[i] = 0.;
    proj_all[i] = 0.;
  }
  for (int i = 0; i < nlocal; i++)
    if (mask[i] & groupbit) {
      proj[0] += f[i][0] * path[i][3]; // f.n
      proj[0] += f[i][1] * path[i][4]; // f.n
      proj[0] += f[i][2] * path[i][5]; // f.n
    }
  if(com_flag == 0){
    c_v[9] += 1.0;
  } else {
    for (int i = 0; i < nlocal; i++)
      if (mask[i] & groupbit) {
        c_v[3] += f[i][0];
        c_v[4] += f[i][1];
        c_v[5] += f[i][2];
        c_v[9] += 1.0;
      }
  }

  MPI_Allreduce(proj,proj_all,5,MPI_DOUBLE,MPI_SUM,world);
  MPI_Allreduce(c_v,c_v_all,10,MPI_DOUBLE,MPI_SUM,world);

  if (od_flag == 0){
    if (rmass) {
      for (int i = 0; i < nlocal; i++)
        if (mask[i] & groupbit) {
          dtfm = dtf / rmass[i];
          v[i][0] += dtfm * (f[i][0]-path[i][3]*proj_all[0] - c_v_all[3]/c_v_all[9]);
          v[i][1] += dtfm * (f[i][1]-path[i][4]*proj_all[0] - c_v_all[4]/c_v_all[9]);
          v[i][2] += dtfm * (f[i][2]-path[i][5]*proj_all[0] - c_v_all[5]/c_v_all[9]);
        }
    } else {
      for (int i = 0; i < nlocal; i++)
        if (mask[i] & groupbit) {
          dtfm = dtf / mass[type[i]];
          v[i][0] += dtfm * (f[i][0]-path[i][3]*proj_all[0] - c_v_all[3]/c_v_all[9]);
          v[i][1] += dtfm * (f[i][1]-path[i][4]*proj_all[0] - c_v_all[4]/c_v_all[9]);
          v[i][2] += dtfm * (f[i][2]-path[i][5]*proj_all[0] - c_v_all[5]/c_v_all[9]);
        }
    }
  } else {
    for (int i = 0; i < nlocal; i++)
      if (mask[i] & groupbit) {
        v[i][0] = 0.;
        v[i][1] = 0.;
        v[i][2] = 0.;
      }
  }
};

/* ---------------------------------------------------------------------- */

void FixPAFITI::initial_integrate_respa(int vflag, int ilevel, int iloop)
{
  dtv = step_respa[ilevel];
  dtf = 0.5 * step_respa[ilevel] * force->ftm2v;

  // innermost level - NVE update of v and x
  // all other levels - NVE update of v

  if (ilevel == 0) initial_integrate(vflag);
  else final_integrate();
};

/* ---------------------------------------------------------------------- */

void FixPAFITI::final_integrate_respa(int ilevel, int iloop)
{
  dtf = 0.5 * step_respa[ilevel] * force->ftm2v;
  final_integrate();
};

/* ---------------------------------------------------------------------- */

void FixPAFITI::reset_dt()
{
  dtv = update->dt;
  dtf = 0.5 * update->dt * force->ftm2v;
};

/* ----------------------------------------------------------------------
   memory usage of local atom-based array
------------------------------------------------------------------------- */

double FixPAFITI::memory_usage()
{
  double bytes = 0.0;
  bytes = maxatom* 3 * sizeof(double);
  // more for TI terms?
  return bytes;
};
