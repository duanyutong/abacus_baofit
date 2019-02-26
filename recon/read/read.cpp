
/*
read.cpp: Ryuichiro Hada, November 2016

This reads N-body simulation data and creats input particles files
(filename and filename2) to be used in "fftcorr.cpp" written by Daniel.

Input particles files have a specific binary format (for the both of
filename and filename2):

--------
64 bytes header:  double posmin[3], posmax[3], max_sep, blank8;
Then 4 doubles per particle: x,y,z,weight, repeating for each particle.

Posmin and posmax are the bounding box for the particle data.
Input posmin and posmax must be oversized so that no periodic replicas are
within max_sep of each other.  Don't forget to include the cell-size
and CIC effects in this estimate; we recommend padding by another
50 [Mpc/h] or so, just to be safe.  This code does not periodic wrap, so
it is required that posmin and posmax be padded.
--------


This code assumes, for now, the followings as the N-body simulation data:

"emulator_1100box_planck_00-0_FoF_halos"
"emulator_720box_planck_00-0_FoF_halos"
"BOSS_1600box_FoF_halos"

*************************

files---

./halos_[x].particles:
    Binary formatted file containing a 10% subsample of particles
    in each halo in ./halos_[x]. Obtaining the particles that correspond
    to a particular halo is described in section 2.3.
./halos_[x].field:
    A 10% subsample of particles not associated with a halo in the slab [x].
    A complete 10% subsample of all particles in the slab can be obtained
    by combining the .field file and the .particles file. An example use of
    this is given in section 3.1.


----[parameters]----

"emulator_1100box_planck_00-0_FoF_halos"
  BoxSize: 1100 [Mpc/h]
  NP: 2985984000 (= 1440^3)  (we use actually 10% subsamples: tot_D = 298578854)
  (posimin, posimax): (-0.5, 0.5)[BoxSize]    (not be padded)

"emulator_720box_planck_00-0_FoF_halos"
  BoxSize: 720 [Mpc/h]
  NP: 2985984000 (= 1440^3)  (we use actually 10% subsamples: tot_D = 298579702)
  (posimin, posimax): (-0.5, 0.5)[BoxSize]    (not be padded)

"BOSS_1600box_FoF_halos"
  BoxSize: 1600 [Mpc/h]
  NP: 32768000000 (= 3200^3)  (we use actually 10% subsamples: tot_D = 3276920257)
  (posimin, posimax): (-0.5, 0.5)[BoxSize]    (not be padded)

**************************

*/


/* ======================= Preamble ================= */
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <sys/stat.h>
#include <complex>
#include <vector>
#include <algorithm>
#include <string>
#include <iostream>
#include <iomanip>
#include <iterator>
#include <sstream>
#include <cstring>
#include <sys/time.h>
#include <omp.h>
#include <random>
#include "STimer.cc"


using namespace std;

typedef unsigned long long int uint64;
typedef unsigned int uint32;

//!!!!!!!! ******* index ********* !!!!!!!!!!

//##### simulation #####
//#define planck         //  for matter or HALO or GAL
  //#define num_phase 0  // [int] phase of simulation
//#define BOSS        //  for matter or HALO
//#define ST

//##### option #####
//#define CHECK                  //  only check
  #define CHECK_num  200      //  (int)  number of check

//#define INITIAL                  //  to get initial density field

//#define SHIFT                    //  to get "true" shift
  //#define initial                 //  to get "initial" displacement
    #define AXIS 2                  //  axis for initial

//#define HALO
  //#define M_cut 12.60206   //  cut_off mass (= 4*10^{12} solar mass)
  //#define M_cut 14       //  cut_off mass
//#define GAL       //  for galaxie (hod)
//#define index_sc 0     //  (int) 0: default, 1: sc = 0.75, 2: sc = -0.75, 3: sc = "du"  

//#define index_rsd              //  to redshift-space

//#define NUM_GRID 480

//##### parameters #####
//#define redshift 0.500           //  (double)  redshift of simulation data

//#define index_ratio_RtoD -1      //  (int)    [-1]  Random = ngird [0~3] Random/Data = [10^0: 0], [10^1: 1] or [10^2: 2]
                                // [INITIAL] [0]   Random/Data =  [2^0: 0] (assumed to be used with periodic option)
//#define index_num_parcentage 3  //  (int)    [0~3] parcentage of particle used = [10^0: 0],[10^(-1): 1],or [10^(-2): 2]....
                                // [INITIAL] [1~3] PPD divided y = [2^1: 1],[2^2: 2],[2^3: 3], or [2^4: 4]
//#define periodic                 //  To use periodic boundary
//#define Random                   //  To compute "Random"
  //#define only_R                 //  To compute only "Random"
    //#define USED_Data 4096000000   //  setting used_D
  //#define Uniform                //  To compute "uniform Random"

//#define no_FKP                   //  not use FKP as weight




//******* input parameter *********

#ifdef planck
  #define BoxSize 1100                 //  (double)  [Mpc/h] (1100 or 720)
  #define CutSize 1100                 //  (double)  [Mpc/h]
  #define H 0.6726                     //  (double)
  #ifndef INITIAL
    #ifdef HALO  // halo
        double total_par_ori_phase[] = { 1812422.0,     //  00-0 for z = 0.5, M_cut 12.60206 (corresponding to "used_D"
                                         2184826.0      //  00-1                               because of n_p = 1 and mass cut)
                                       };
                                       double total_par_ori = total_par_ori_phase[num_phase];
        #define posmin_ori -0.5        //  (double)  [BoxSize]    (not be padded)
        #define posmax_ori 0.5
        #define file_number 7               //  (int)  the number of files
    #else
      #ifdef GAL // gal
                                       double total_par_ori_phase[][50]  = {
        // index_sc = 0; sc = 0.0 (default)                                  
                                   { 586502.0,     //  00-0 for z = 0.5
                                     585533.0,    //  00-1 for z = 0.5
                                     586799.0,    //  00-2 for z = 0.5
                                     585269.0,    //  00-3 for z = 0.5
                                     585561.0,    //  00-4 for z = 0.5
                                     585528.0,    //  00-5 for z = 0.5
                                     585407.0,    //  00-6 for z = 0.5
                                     585370.0,    //  00-7 for z = 0.5
                                     586459.0,    //  00-8 for z = 0.5
                                     586203.0,    //  00-9 for z = 0.5
                                     585353.0,    //  00-10 for z = 0.5
                                     586037.0,    //  00-11 for z = 0.5
                                     584566.0,    //  00-12 for z = 0.5
                                     586409.0,    //  00-13 for z = 0.5
                                     585589.0,    //  00-14 for z = 0.5
                                     585908.0     //  00-15 for z = 0.5
                                   },
        // index_sc = 1; sc = 0.75
                                   { 55760.0,    //  00-0 for z = 0.5
                                     55760.0,    //  00-1 for z = 0.5
                                     55470.0,    //  00-2 for z = 0.5
                                     55594.0,    //  00-3 for z = 0.5
                                     55799.0,    //  00-4 for z = 0.5
                                     55437.0,    //  00-5 for z = 0.5
                                     55307.0,    //  00-6 for z = 0.5
                                     55770.0,    //  00-7 for z = 0.5
                                     55632.0,    //  00-8 for z = 0.5
                                     55539.0,    //  00-9 for z = 0.5
                                     55862.0,    //  00-10 for z = 0.5
                                     55956.0,    //  00-11 for z = 0.5
                                     55734.0,    //  00-12 for z = 0.5
                                     55944.0,    //  00-13 for z = 0.5
                                     55594.0,    //  00-14 for z = 0.5
                                     56100.0     //  00-15 for z = 0.5
                                   },
        // index_sc = 2; sc = -0.75
                                   { 3496658.0,    //  00-0 for z = 0.5
                                     3496658.0,    //  00-1 for z = 0.5
                                     3496814.0,    //  00-2 for z = 0.5
                                     3492391.0,    //  00-3 for z = 0.5
                                     3494351.0,    //  00-4 for z = 0.5
                                     3493519.0,    //  00-5 for z = 0.5
                                     3494677.0,    //  00-6 for z = 0.5
                                     3494268.0,    //  00-7 for z = 0.5
                                     3495643.0,    //  00-8 for z = 0.5
                                     3498026.0,    //  00-9 for z = 0.5
                                     3494931.0,    //  00-10 for z = 0.5
                                     3496585.0,    //  00-11 for z = 0.5
                                     3492535.0,    //  00-12 for z = 0.5
                                     3494844.0,    //  00-13 for z = 0.5
                                     3495116.0,    //  00-14 for z = 0.5
                                     3496333.0     //  00-15 for z = 0.5
                                   },
        // index_sc = 3; sc = "du"
                                   { 475283.0,    //  00-0 for z = 0.5
                                  // 573225.0    //  00-1 for z = 0.5  in real
                                     572396.0    //  00-1 for z = 0.5  in redshift 
                                   }
                                 };
                                 double total_par_ori = total_par_ori_phase[index_sc][num_phase];
        #define posmin_ori 0.0        //  (double)  [BoxSize]    (not be padded)
        #define posmax_ori 1.0
        #define file_number 1               //  (int)  the number of files
      #else
        // matter
        double total_par_ori_phase[] = { 298596066.0   //  00-0~15,  10% (from 1440^3) subsamples (corresponding to tot_D)
        };
              // this number is not known originally, so we need to run Data part only to get tot_D
        double total_par_ori = total_par_ori_phase[0];
        #define posmin_ori -0.5        //  (double)  [BoxSize]    (not be padded)
        #define posmax_ori 0.5
        #define file_number 7               //  (int)  the number of files
      #endif
    #endif
  #else
    double total_par_ori = 2985984000.0;  //  (double) (= 1440^3) "full"samples
    #define posmin_ori -0.5        //  (double)  [BoxSize]    (not be padded)
    #define posmax_ori 0.5
  #endif
  #define file_number_ini 375
  #define PPD 1440
  #define MP 5.776643562630781e+10    //  (double)  [solar mass] particle mass
#endif

#ifdef BOSS
  #define BoxSize 1600                //  (double)  [Mpc/h]
  #define CutSize 1600                 //  (double)  [Mpc/h]
  #define H 0.6726                     //  (double)
  #ifndef INITIAL
    #ifdef HALO // halo
      // double total_par_ori = 59961582.0;     //  (double) for z = 0.5
      double total_par_ori = 57246751.0;     //  (double) for z = 1.0
    #else // matter
      double total_par_ori = 3276920257.0;   //  (double)  10% (from 3200^3) (corresponding to tot_D)
            // this number is not known originally, so we need to run Data part only to get tot_D
    #endif
  #else
      double total_par_ori = 32768000000.0;  //  (double) (= 3200^3) "full"samples
  #endif
  #define file_number 219              //  (int)  the number of files
  #define file_number_ini 875
  #define PPD 3200
  #define MP 5.776643562630781e+10    //  (double)  [solar mass] particle mass
  #define posmin_ori -0.5        //  (double)  [BoxSize]    (not be padded)
  #define posmax_ori 0.5
#endif

#ifdef ST
  #define BoxSize 4000                 //  (double)  [Mpc/h]
  #define CutSize 1600                 //  (double)  [Mpc/h]
  #define H 0.71                     //  (double)
  #ifdef HALO
  //double total_par_ori = 77395569.0;     //  (double) for m001 all
    //double total_par_ori = 4950185.0;    //  (double) for m001, CutSize 1600
      //double total_par_ori = 4257538.0;  //  (double) for m001, CutSize 1600, M_cut 12.5
      //double total_par_ori = 1337601.0;  //  (double) for m001, CutSize 1600, M_cut 13
      //double total_par_ori = 344792.0;  //  (double) for m001, CutSize 1600, M_cut 13.5
      double total_par_ori = 65105.0;  //  (double) for m001, CutSize 1600, M_cut 14
  //double total_par_ori = 77417680.0;     //  (double) for m002
  //double total_par_ori = 77407545.0;     //  (double) for m003
  //double total_par_ori = 77427693.0;     //  (double) for m004
  #endif
  #define file_number 1              //  (int)  the number of files
  #define PPD 3200                   //   !!!!  fake  !!!!
  #define posmin_ori 0.0        //  (double)  [BoxSize]    (not be padded)
  #define posmax_ori 1.0
#endif

// Buffer
#define BUFFERSIZE 256         //  I'm not sure if this is effective or not...

//#define MY_SCHEDULE schedule(dynamic,256)
#define MY_SCHEDULE schedule(static,1)

// Class for Clock
STimer Data_t, Data_ini_t, Random_t, vec_set_t;



//******* Structure *********

//  structure for header
typedef struct
      {
    double posmin[3];   // [Mpc/h]
    double posmax[3];   // [Mpc/h]
    double maxcep;      // [Mpc/h]
    double blank8;
  } __attribute__((__packed__)) struct_header;


//  structure for particles input
typedef struct
  {
    float x[3];         // [Mpc/h] for planck,   [Mpc/h] for BOSS
    float v[3];         // [km/s] for planck,   [km/s] for BOSS
  } __attribute__((__packed__)) struct_in_x3v3;


#if defined (planck) || defined (BOSS)
  typedef struct
  {
    uint64 id;
    uint64 subsamp_start;
    uint32 subsamp_np;
    uint32 N;
    uint32 subhalo_N[4];
      float x[3];         // [Mpc/h]
      float v[3];         // [km/s]
      float sigma_v[3];
      float r25, r50, r75, r90;
      float vcirc_max, rvcirc_max;
      float subhalo_x[3];         // [Mpc/h]
      float subhalo_v[3];         // [km/s]
      float subhalo_sigma_v[3];
      float subhalo_r25, subhalo_r50, subhalo_r75, subhalo_r90;
      float subhalo_vcirc_max, subhalo_rvcirc_max;
    } __attribute__((__packed__)) struct_in_halo;
#endif
#ifdef ST
    typedef struct
    {
    float m;            // [solar mass]
    float x[3];         // [Mpc/h]
    float v[3];         // [km/s]
  } __attribute__((__packed__)) struct_in_halo;
#endif


//  structure for gals input
#ifdef planck
  typedef struct
  {
      float m;            // [solar mass]
      float x[3];         // [Mpc]
    } __attribute__((__packed__)) struct_in_gal;
#endif


//  structure for ID input
typedef struct
{
    uint64 id;         // []
} __attribute__((__packed__)) struct_in_id;

//  class for initial data input (ICFormat = "RVZel")
class RVZelParticle {
  public:
    unsigned short i,j,k;
    float displ[3];
    float vel[3];
  };


//  structure for data output  //
  typedef struct
  {
    double x[3];        // [Mpc/h]
    double w;
  } __attribute__((__packed__)) struct_out_x3w;

// recursive mkdir
  static void _mkdir(const char *dir)
  {
    char tmp[256];
    char *p = NULL;
    size_t len;
    snprintf(tmp, sizeof(tmp),"%s",dir);
    len = strlen(tmp);
    if(tmp[len - 1] == '/')
      tmp[len - 1] = 0;
    for(p = tmp + 1; *p; p++)
      if(*p == '/') {
        *p = 0;
        mkdir(tmp, S_IRWXU);
        *p = '/';
      }
      mkdir(tmp, S_IRWXU);
    }


//  definition: long int   //

    long int num_par, num_fie, num_halo, num_ini, num_ran;
    long int num_par_id, num_fie_id, num_halo_id;
    long int sum_num_par = 0;
    long int sum_num_used_par = 0;
    long int sum_num_fie = 0;
    long int sum_num_used_fie = 0;
    long int sum_num_halo = 0;
    long int sum_num_used_halo = 0;
    long int sum_num_ini = 0;
    long int sum_num_used_ini = 0;
    long int sum_num = 0;
    long int sum_num_used = 0;
    long int tot_par, tot_fie, tot_halo, tot_ini, tot_D, used_D, tot_R, tot_R_p;
    long int num_loop_p = 0;
    long int num_loop_f = 0;
    long int num_loop_h = 0;
    long int num_loop_i = 0;
    long int num_loop_R = 0;
    uint64 PIDt, ix_ori, iy_ori, iz_ori, num_file_ini, num_file_ini_tem, num_blo, num_sub_blo, num_res;

// merge sort definition
    vector<uint64> merge(vector<uint64> left, vector<uint64> right);
    vector<uint64> mergeSort(vector<uint64> m);



//******* Main *********

    int main(int argc, char *argv[])
    {
      long int ngrid_1d;
      ngrid_1d = NUM_GRID;

    #ifdef inp_total_num
      total_par_ori = (double)total_par_ori_inp; 
    #endif

    //******* Path *********

    // Boxsize: int >> string
      stringstream sbs;
      sbs << (int)BoxSize;
      string p_boxsize = sbs.str();

    #ifdef HALO
      // M_cut: float >> string
      stringstream s3;
      s3 << (float)M_cut;
      string M_cut_s = s3.str();
    #endif
    // redshift: float >> string
      stringstream s4;
      s4 << fixed << redshift;
      string redshift_s = s4.str();
      redshift_s.erase(redshift_s.begin() + 5, redshift_s.begin() + 8);

    // ngrid_1D: int >> string
      stringstream ngs;
      ngs << ngrid_1d;
      string ngrid_1D_s = ngs.str();

    // index_ratio_RtoD: int >> string
      stringstream s1;
      s1 << index_ratio_RtoD;
      string ind_RtoD = s1.str();

    // index_num_parcentage: int >> string
      stringstream s2;
      s2 << index_num_parcentage;
      string ind_parc_used = s2.str();

      stringstream s3;
      s3 << tag;
      string ftag = s3.str();

    // ******* general ********
      string p_rsd = "/real";
      string ind_FKP  = "_F";
      string ind_uni  = "";
      string ind_peri = "";
      string ind_shift= "";
    #ifdef index_rsd
      p_rsd = "/rsd";
    #endif
    #ifdef no_FKP
      ind_FKP = "_noF";
    #endif
    #ifdef Uniform
      ind_uni = "_uni";
    #endif
    #ifdef periodic
      ind_peri = "_p";
    #endif
    #ifdef SHIFT
      ind_shift = "_S";
    #endif

    // ******* each simulation ********
    #ifdef planck
      // path to emulator files
      // num_phase: int >> string
      stringstream nphase;
      nphase << (int)num_phase;
      string phase = nphase.str();

      string p_to_public = "/mnt/store2/bigsim_products";     // public
      string p_to_me = "/mnt/store1/rhada";       // me
      string p_typesim = "/AbacusCosmos";
      string p_project = "/emulator_" + p_boxsize + "box_planck_products";
      string p_phase   = "/emulator_" + p_boxsize + "box_planck_00-" + phase + "_products";
      string p_typehalo= "/emulator_" + p_boxsize + "box_planck_00-" + phase + "_FoF_halos";
      string p_z = "/z" + redshift_s;
      string p_sim = p_project + p_phase + p_typehalo;
      string p_sample     = "/fin_" + ind_RtoD + "_" + ind_parc_used
      + ind_FKP + "_ng" + ngrid_1D_s + ind_uni + ind_peri + ind_shift;
      string p_sample_ini = "/ini_" + ind_RtoD + "_" + ind_parc_used
      + ind_FKP + "_ng" + ngrid_1D_s + ind_uni + ind_peri;
      #ifndef GAL
        #ifndef HALO
        // matter
      string in_path  = p_to_public + p_sim + p_z;
      string out_path = p_to_me + p_typesim + p_sim + p_z + "/matter";
        #else
        // halo
      string in_path = p_to_public + p_sim + p_z;
      string out_path = p_to_me + p_typesim + p_sim + p_z + "/halo_Mc" + M_cut_s;
        #endif
      #else
        // gal
        string p_op = "";     // default
        if(index_sc == 1){
          p_op = "_sc0.750";
        }
        else if(index_sc == 2){
          p_op = "_sc-0.750";
        }
        else if(index_sc == 3){
          p_op = "_du";
        }
        string in_path = p_to_me + p_typesim + p_sim + p_z;
        string out_path = p_to_me + p_typesim + p_sim + p_z + "/gal" + p_op;
      #endif
      out_path = "/home/dyt/store/recon/temp/"; //out_path += p_sample + p_rsd;
      // initial
      string in_path_ini = p_to_me + p_typesim + p_sim + "/ini/IC/scratch";
      // #ifdef INITIAL
      //   out_path = p_to_me + p_typesim + p_sim + "/ini" + p_sample_ini;
      // #endif
      in_path = "/home/dyt/store/recon/temp/"; //in_path += "/";
      in_path_ini += "/";
    #endif

    #ifdef BOSS
      //  path to emulator files
      string p_to_emulator = "/mnt/store1/bigsim_products/";
      string p_project = "BOSS_" + p_boxsize + "box_";
      string p_product = "products/";
      string p_phase = "";
      string p_what = "FoF_halos";
      string p_z = "/z" + redshift_s;
      #ifdef INITIAL
      p_z = "ini";
      #endif
      string in_path = p_to_emulator + p_project + p_product
      + p_project + p_what + "/" + p_z + "/";

      string in_path_ini = "/mnt/store1/rhada/" + p_project + p_phase + p_what + "/ini/";

      //  path to output files
      string out_path = "/mnt/store1/rhada/" + p_project + p_phase + p_what + "/" + p_z + "/";
    #endif

    #ifdef ST
      //  path to emulator files
      string p_to_emulator = "/mnt/store1/rhada/ST/";
      string p_project = "halo";
      string p_phase = "m001";
      string p_what = "_fofproperties";
      string p_z = "/z" + redshift_s;

      string in_path = p_to_emulator + p_project + "/";

      string in_path_ini = in_path;

      //  path to output files
      string out_path = p_to_emulator + p_project + "/";
    #endif

//out_path = "file";

    // doc. files: definition
      string docfile_str = out_path + "/";
    #ifdef CHECK
      docfile_str += "Check_";
    #endif
      docfile_str += "doc_file_";
    #ifdef only_R
      docfile_str += "Ronly";
    #else
      docfile_str += "DandR";
    #endif
    //docfile_str += "_test";
      docfile_str += ".txt";

    // out_path: string >> char
      char* out_path_c = new char[out_path.size() + 1];
      strcpy(out_path_c,out_path.c_str());

    // doc. files: string >> char
      char* docfile = new char[docfile_str.size() + 1];
      strcpy(docfile,docfile_str.c_str());

      struct stat stat_out_path;
      struct stat stat_docfile;
      string ind_doc;
    // if(stat(out_path_c, &stat_out_path) != 0) {
    //     fprintf(stderr,"# %s\n# doesn't exist\n", out_path_c);
    //     _mkdir(out_path_c);
    //     fprintf(stderr,"# has been produced\n");
    // }
    // if(stat(docfile, &stat_docfile) == 0){
    //     fprintf(stderr,"# This docfile has already existed !!\n");
    //     cout << "# Do you really want to run ?>>\n";
    //     cin  >> ind_doc;
    //     if(ind_doc != "y") exit(0);
    // }

    // doc. files: docfile << stdout
      FILE *discard=freopen(docfile,"w", stdout);
      assert(discard!=NULL&&stdout!=NULL);

    // output files: definition
      string out_filename_D = out_path;
      out_filename_D += "/file_D-" + ftag;
    //out_filename_D += "_test";

      string out_filename_D_S0 = out_filename_D + "_S0";
      string out_filename_D_S1 = out_filename_D + "_S1";
      string out_filename_D_S2 = out_filename_D + "_S2";

      string out_filename_D_ini = out_filename_D;
      out_filename_D_ini += "_ini";

      string out_filename_R = out_path;
      out_filename_R += "/file_R-" + ftag;
    //out_filename_R += "_test";

    // output files: string >> char
      char* out_D = new char[out_filename_D.size() + 1];
      strcpy(out_D,out_filename_D.c_str());

    #ifdef SHIFT
      char* out_D_S0 = new char[out_filename_D_S0.size() + 1];
      strcpy(out_D_S0,out_filename_D_S0.c_str());

      char* out_D_S1 = new char[out_filename_D_S1.size() + 1];
      strcpy(out_D_S1,out_filename_D_S1.c_str());

      char* out_D_S2 = new char[out_filename_D_S2.size() + 1];
      strcpy(out_D_S2,out_filename_D_S2.c_str());

      char* out_D_ini = new char[out_filename_D_ini.size() + 1];
      strcpy(out_D_ini,out_filename_D_ini.c_str());
    #endif

      char* out_R = new char[out_filename_R.size() + 1];
      strcpy(out_R,out_filename_R.c_str());

      char* outdir = new char[out_path.size() + 1];
      strcpy(outdir,out_path.c_str());

    // header: definition for structure
      struct_header header;

    // header: computation of header inclding padding
      for (int p=0; p<3; p++) {
        header.posmin[p] = (double)(-0.5*CutSize);
        header.posmax[p] = (double)(0.5*CutSize);
        #ifndef periodic
        header.posmin[p] -= 50.0;
           // another 50[Mpc/h] taking account of shift due to reconstruction and RSD
        header.posmax[p] += maxcep_ori + 50.0;
           // another 50[Mpc/h]
        #endif
      }
      header.maxcep = (double)(maxcep_ori);
    header.blank8 = 123.456;  // 8byte blank


    double n_p_d;
    int n_p_one, n_p;
    n_p_d = pow(10,index_num_parcentage);

    if(index_num_parcentage == 0) n_p_d = 1.0;
    else if(index_num_parcentage == 1) n_p_d = (double)total_par_ori/pow(800, 3);
    else if(index_num_parcentage == 2) n_p_d = (double)total_par_ori/pow(640, 3);
    else if(index_num_parcentage == 3) n_p_d = (double)total_par_ori/pow(480, 3);
    else if(index_num_parcentage == 4) n_p_d = (double)total_par_ori/pow(200, 3);
    if ((n_p_d - floor(n_p_d)) < 0.5) n_p = floor(n_p_d);
    else n_p = ceil(n_p_d);

    #ifdef INITIAL                                      // BOSS      ,planck
      if(index_num_parcentage == 4) n_p_one = 16;       // 200^3
      else if(index_num_parcentage == 3)   n_p_one = 8;   // 400^3
      else if(index_num_parcentage == 2)   n_p_one = 5;   // 640^3
      else if(index_num_parcentage == 1.5) n_p_one = 4;   //           ,360^3
      else if(index_num_parcentage == 1)   n_p_one = 3;   //           ,480^3
      else if(index_num_parcentage == 0.5) n_p_one = 2;   //           ,720^3
      //n_p_one = 1;
      n_p_d = pow(n_p_one, 3);
      n_p = ceil(n_p_d);
    #endif

      if(n_p <= 1) n_p = 1;

      fprintf(stderr,"# n_p_one = %d\n",n_p_one);
      fprintf(stderr,"# n_p = %d\n",n_p);
      fprintf(stderr,"# total_par_ori = %e\n",total_par_ori);

      int axis;
      double ra;
      if(index_ratio_RtoD == -1){
        ra = pow(ngrid_1d,3)/(total_par_ori/n_p);
      }
      else{
        ra = pow(10,index_ratio_RtoD);
      }
    #ifdef INITIAL
      if(index_ratio_RtoD == 0) ra = 1.0;
    #endif
      axis = (int)AXIS;

      if((total_par_ori/n_p*ra) < pow(ngrid_1d,3)){
        fprintf(stderr,"# Error should happen !!!!\n");
      }

      long int B;
      B = BUFFERSIZE*100;
    // if index_num_parcentage > 2, error happens. So need to reduce BUFFERSIZE

    // definition: L
      double L, C, U_x, U_v, h;
      L = (double)BoxSize;
      C = (double)CutSize;
      h = (double)H;
    #ifdef planck
      #ifndef GAL // matter & halo
        U_x = 1.0;   // position * U_x [Mpc/h]
        if(redshift == 0.5){
        //(1 + z)/(Hubblenow*100[km/s/(Mpc/h)])  [Hubblenow = H(z)/H_0]
          U_v = (1 + 0.500)/(1.321406 * 100.0);    // velocity * U_v * U_x [Mpc/h]
          //fprintf(stderr,"U_v = %f\n", U_v);
        }
        else if(redshift == 0.7){
        //(1 + z)/(Hubblenow*100[km/s/(Mpc/h)])  [Hubblenow = H(z)/H_0]
          U_v = (1 + 0.700)/(1.493077 * 100.0);    // velocity * U_v * U_x [Mpc/h]
          //fprintf(stderr,"U_v = %f\n", U_v);
        }
        else{
          fprintf(stderr,"Define U_v for new redshift !!\n");
        }
      #else // GRAND-HOD
        // U_x = h;     // position * U_x [Mpc/h]
        // if(index_sc == 3){
        U_x = 1.0;
        // }
        U_v = 1.0;   // velocity * U_v * U_x [Mpc/h]
      #endif
    #endif
    #ifdef BOSS
      U_x = 1.0;   // position * U_x [Mpc/h]
      if(redshift == 0.5){
      //(1 + z)/(Hubblenow*100[km/s/(Mpc/h)])  [Hubblenow = H(z)/H_0]
        U_v = (1 + 0.500)/(1.321406 * 100.0);    // velocity * U_v * U_x [Mpc/h]
        //fprintf(stderr,"U_v = %f\n", U_v);
      }
      else if(redshift == 1.0){
      //(1 + z)/(Hubblenow*100[km/s/(Mpc/h)])  [Hubblenow = H(z)/H_0]
        U_v = (1 + 1.000)/(1.788595 * 100.0);    // velocity * U_v * U_x [Mpc/h]
        //fprintf(stderr,"U_v = %f\n", U_v);
      }
      else{
        fprintf(stderr,"Define U_v for new redshift !!\n");
      }
    #endif
    #ifdef ST
      U_x = 1.0;     // position * U_x [Mpc/h]
      if(redshift == 0.15){
      //(1 + z)/(Hubblenow*100[km/s/(Mpc/h)])  [Hubblenow = H(z)/H_0]
        U_v = (1 + 0.150)/(1.066737 * 100.0);    // velocity * U_v * U_x [Mpc/h]
        //fprintf(stderr,"U_v = %f\n", U_v);
      }
      else{
        fprintf(stderr,"Define U_v for new redshift !!\n");
      }
    #endif

    // definition for "temporary" alpha
      double alpha;
      alpha = 1./ra;

    // computing the weight
    double n_ave; // [Mpc/h]^(-3) average number density denpending on redshift
    n_ave = (total_par_ori/n_p)/pow(C,3);
    double weight;
    #ifndef no_FKP
    weight = 1.0/((1.0 + alpha) + n_ave*power_0);
    #else
    weight = 1.0/(n_ave*power_0);
    #endif

    // variables: definition
    double p_max, p_min, p_max_D, p_min_D, ppd, p_max_a, p_min_a;
    uint64 ppd_int;
    p_min_D = (double)posmin_ori;
    p_max_D = (double)posmax_ori;
    p_min = -0.5;
    p_max = 0.5;
    ppd = (double)PPD;
    ppd_int = (uint64)PPD;
    p_max_a = 0.0;
    p_min_a = 0.0;

    //OpenMP: set the number of threads
    omp_set_num_threads(10);    // limit the number of threads for alan and gosling
    fprintf(stdout,"# Running with %d threads\n", omp_get_max_threads());
    fprintf(stdout,"# This is in %s\n\n", outdir);

#ifndef CHECK

  #ifndef only_R

    #ifndef INITIAL
      // input: definition for structure
      #ifdef HALO  // halo
    struct_in_halo x_v[B];
      #else
        #ifdef GAL // gal
    struct_in_gal x_v[B];
        #else // matter
    struct_in_gal x_v[B];  // struct_in_x3v3 x_v[B];
        #endif
      #endif
    fprintf(stderr,"# Byte of x_v[]: %ld\n", sizeof(x_v[0]));
      #ifdef SHIFT
    struct_in_id ID[B];
    RVZelParticle x_ini[B];
    uint64 sum_par = 819230229;
        //vector<uint64> PID(sum_par);
    vector<uint64> PID;
      #endif
    #else
      // input: definition for class
    RVZelParticle x_ini[B];
    fprintf(stderr,"# Byte of x_ini[]: %ld\n", sizeof(x_ini[0]));
    #endif

    // output: definition for structure
    struct_out_x3w x_w[B];
    #ifdef SHIFT
    struct_out_x3w x_s0[B];
    struct_out_x3w x_s1[B];
    struct_out_x3w x_s2[B];
    #endif

    // non-deterministic random number generator
    random_device rnds;
    // Mersenne Twister, 64bit version
    mt19937_64 mts(rnds());

//#ifndef SHIFT
    //******* Data (>>filename) *********

    Data_t.Start();
    // itarating "num_files" times for all files halos_[x]
    int num_files;
    #ifndef INITIAL
    num_files = file_number;
    #else
    num_files = file_number_ini;
    #endif
    for(int j=0; j<num_files; j++){

      // j: int >> string
      stringstream ss;
      ss << j;
      string num_file = ss.str();

      // input files: definition
      #ifdef planck
      string f_p = "recon_array-" + ftag + ".dat"; //string f_p = "particles_";
      string f_p_id = "particle_ids_";
      string f_f = "field_particles_";
      string f_f_id = "field_ids_";
      string l_p = "";
      string l_f = "";
        #ifdef HALO
      string f_h = "halos_";
      string l_h = "";
        #endif
        #ifdef GAL
          #ifdef index_rsd
      string index_rsd_st = "_rsd";
          #else
      string index_rsd_st = "";
          #endif
          string f_h = "recon_array-" + ftag + ".dat"; // string f_h = "GRAND_HOD" + p_op + index_rsd_st;
          string l_h = "";
          num_file = "";
        #endif
      #endif

      #ifdef BOSS
          string f_p = "particles_";
          string f_p_id = "particle_ids_";
          string f_f = "field_particles_";
          string f_f_id = "field_ids_";
          string f_h = "halos_";
          string l_p = "";
          string l_f = "";
          string l_h = "";
      #endif

      #ifdef ST
          string f_p = "";
          string f_p_id = "";
          string f_f = "";
          string f_f_id = "";
        #ifdef HALO
          string f_h = p_phase + p_what + "_" + p_z + "_halo";
        #endif
          string l_p = "";
          string l_f = "";
          string l_h = "";
          num_file = "";
      #endif

          // particles (in halos)
          string in_file_par = in_path + f_p + num_file + l_p;
          string in_file_name_par = f_p + num_file + l_p;

          // field particle (in field)
          string in_file_fie = in_path + f_f + num_file + l_f;
          string in_file_name_fie = f_f + num_file + l_f;

        #if defined(HALO) || defined(GAL)
          // halos
          string in_file_halo = in_path + f_h + num_file + l_h;
          string in_file_name_halo = f_h + num_file + l_h;
        #endif

        #ifdef SHIFT
          // particle ID
          string in_file_par_id = in_path + f_p_id + num_file;
          in_file_name_par += "+ids";

          // field ID
          string in_file_fie_id = in_path + f_f_id + num_file;
          in_file_name_fie += "+ids";
        #endif

          // initial
          string in_file_ini = in_path_ini + "ic_" + num_file;
          string in_file_name_ini = "ic_" + num_file;


          FILE *fp_out_D, *fp_out_D_S0, *fp_out_D_S1, *fp_out_D_S2;
          if(j == 0){
        fp_out_D = fopen(out_D, "wb");  // from the beginning（j＝0）
        assert(fp_out_D!=NULL);
        // output: header
        fwrite(&header, sizeof(header), 1, fp_out_D);
        #ifdef SHIFT
          fp_out_D_S0 = fopen(out_D_S0, "wb");  // from the beginning（j＝0）
          assert(fp_out_D_S0!=NULL);
          // output: header
          fwrite(&header, sizeof(header), 1, fp_out_D_S0);
          fp_out_D_S1 = fopen(out_D_S1, "wb");  // from the beginning（j＝0）
          assert(fp_out_D_S1!=NULL);
          // output: header
          fwrite(&header, sizeof(header), 1, fp_out_D_S1);
          fp_out_D_S2 = fopen(out_D_S2, "wb");  // from the beginning（j＝0）
          assert(fp_out_D_S2!=NULL);
          // output: header
          fwrite(&header, sizeof(header), 1, fp_out_D_S2);
        #endif
        }
        else{
        fp_out_D = fopen(out_D, "ab");  // from the middle (j>=1)
        assert(fp_out_D!=NULL);
        #ifdef SHIFT
          fp_out_D_S0 = fopen(out_D_S0, "ab");  // from the beginning (j>=1)
          assert(fp_out_D_S0!=NULL);

          fp_out_D_S1 = fopen(out_D_S1, "ab");  // from the beginning (j>=1)
          assert(fp_out_D_S1!=NULL);

          fp_out_D_S2 = fopen(out_D_S2, "ab");  // from the beginning (j>=1)
          assert(fp_out_D_S2!=NULL);
        #endif
        }

    #ifndef INITIAL
      #if !defined(HALO) && !defined(GAL)

        //******* .particles *********

          // input files(.particles): string >> char
        char* particles = new char[in_file_par.size() + 1];
        strcpy(particles,in_file_par.c_str());

        FILE *fp_in_par, *fp_in_par_id;
        fp_in_par = fopen(particles, "rb");
        assert(fp_in_par!=NULL);  // problem

          #ifdef SHIFT
            // input files(.particle_ids): string >> char
        char* particle_id = new char[in_file_par_id.size() + 1];
        strcpy(particle_id,in_file_par_id.c_str());

        fp_in_par_id = fopen(particle_id, "rb");
        assert(fp_in_par_id!=NULL);
          #endif

          // name of input files(.particles): string >> char
        char* name_particles = new char[in_file_name_par.size() + 1];
        strcpy(name_particles,in_file_name_par.c_str());

        fprintf(stdout,"# Now, reading file: %s, No. %d\n",name_particles, j);

        num_par = B;
        while (num_par == B) {
            // reading .particles with BUFFERSIZE series into structure x_v[]
          num_par = fread(x_v, sizeof(x_v[0]), B, fp_in_par);
            #ifdef SHIFT
          num_par_id = fread(ID, sizeof(ID[0]), B, fp_in_par_id);
          assert(num_par == num_par_id);
            #endif
          vector<long int> k_v(num_par);
          iota(k_v.begin(), k_v.end(), 0);
            // shuffle vector k_v
          shuffle(k_v.begin(), k_v.end(), mts);
          int j = 0;
          long int k = 0;
          int z_off = 1;
            //#pragma omp parallel for MY_SCHEDULE
          for(int i=0; i<num_par; i+=n_p)
          {
                // uniform random number in the rage of [0, num_par-1)]
                //uniform_int_distribution<> randp(0, (num_par-1));
                //k = randp(mts);

            k = k_v[i];
                //moving and castig x_v[] and weight to x_w[]
            x_w[j].x[0] = (double)(x_v[k].x[0]*U_x) + (-0.5 - p_min_D)*L;
            x_w[j].x[1] = (double)(x_v[k].x[1]*U_x) + (-0.5 - p_min_D)*L;
                #ifdef index_rsd
            x_w[j].x[2] = (double)((x_v[k].x[2] + x_v[k].v[2]*U_v)*U_x) + (-0.5 - p_min_D)*L;
                #else
            x_w[j].x[2] = (double)(x_v[k].x[2]*U_x) + (-0.5 - p_min_D)*L;
                #endif
            x_w[j].w = weight;
                #ifdef periodic
            if(x_w[j].x[0] < p_min*L) x_w[j].x[0] += L;
            if(x_w[j].x[0] >= p_max*L) x_w[j].x[0] -= L;
            if(x_w[j].x[1] < p_min*L) x_w[j].x[1] += L;
            if(x_w[j].x[1] >= p_max*L) x_w[j].x[1] -= L;
            if(x_w[j].x[2] < p_min*L) x_w[j].x[2] += L;
            if(x_w[j].x[2] >= p_max*L) x_w[j].x[2] -= L;
                #endif
                #if defined SHIFT
                    //k = i;
            PIDt = ID[k].id;
            while(z_off){
              iz_ori = PIDt%ppd_int;
              iy_ori = ((PIDt - iz_ori)/ppd_int)%ppd_int;
              ix_ori = (((PIDt - iz_ori)/ppd_int) - iy_ori)/ppd_int;
              x_s0[j].x[0] = (double)(ix_ori)*L/ppd - L/2.0;
              x_s0[j].x[1] = (double)(iy_ori)*L/ppd - L/2.0;
              x_s0[j].x[2] = (double)(iz_ori)*L/ppd - L/2.0;
                    //  fprintf(stderr,"= %e\n",x_s0[j].x[0]);
              assert(x_s0[j].x[0] >= - 0.5*L && x_s0[j].x[0] < 0.5*L);
              assert(x_s0[j].x[1] >= - 0.5*L && x_s0[j].x[1] < 0.5*L);
              assert(x_s0[j].x[2] >= - 0.5*L && x_s0[j].x[2] < 0.5*L);
              x_s1[j].x[0] = x_s0[j].x[0];
              x_s2[j].x[0] = x_s0[j].x[0];
              x_s1[j].x[1] = x_s0[j].x[1];
              x_s2[j].x[1] = x_s0[j].x[1];
              x_s1[j].x[2] = x_s0[j].x[2];
              x_s2[j].x[2] = x_s0[j].x[2];

              x_s0[j].w = (double)(x_v[k].x[0]*U_x) - x_s0[j].x[0];
              x_s1[j].w = (double)(x_v[k].x[1]*U_x) - x_s1[j].x[1];
                    #if defined(index_rsd)
              x_s2[j].w = (double)((x_v[k].x[2] + x_v[k].v[2]*U_v)*U_x) - x_s2[j].x[2];
                    #else
              x_s2[j].w = (double)(x_v[k].x[2]*U_x) - x_s2[j].x[2];
                    #endif
              if(x_s0[j].w < - 0.5*L) x_s0[j].w += L;
              if(x_s0[j].w >= 0.5*L) x_s0[j].w -= L;
              if(x_s1[j].w < - 0.5*L) x_s1[j].w += L;
              if(x_s1[j].w >= 0.5*L) x_s1[j].w -= L;
              if(x_s2[j].w < - 0.5*L) x_s2[j].w += L;
              if(x_s2[j].w >= 0.5*L) x_s2[j].w -= L;
                    /*
                    fprintf(stderr,"par_tru:  ID = %lld, x_ini = %e, y_ini = %e, z_ini = %e\n",
                                PIDt, x_w[j].x[0], x_w[j].x[1], x_w[j].x[2]);
                    fprintf(stderr,"par_tru:  x = %e, y = %e, z = %e, shift = %e\n",
                                x_v[k].x[0], x_v[k].x[1], x_v[k].x[2], x_w[j].w);
                    fprintf(stderr,"par_tru:  v_x = %e, v_y = %e, v_z = %e\n",
                                x_v[k].v[0]*U_v, x_v[k].v[1]*U_v, x_v[k].v[2]*U_v);
                    */
              if(x_s2[j].w < -80.0){
                PIDt -= 256;
              }
              else break;
            }
            PID.push_back(PIDt);
                #endif
            j++;
          }
            // outputting structure x_w[] to file_D
          fwrite(x_w, sizeof(x_w[0]), j, fp_out_D);
            #if defined SHIFT
          fwrite(x_s0, sizeof(x_s0[0]), j, fp_out_D_S0);
          fwrite(x_s1, sizeof(x_s1[0]), j, fp_out_D_S1);
          fwrite(x_s2, sizeof(x_s2[0]), j, fp_out_D_S2);
            #endif
          sum_num_par += num_par;
            sum_num_used_par += j;//floor(num_par/n_p);
            num_loop_p += 1;
          }
          fclose(fp_in_par);


        //******* .field *********

          // input files(.field): string >> char
          char* field = new char[in_file_fie.size() + 1];
          strcpy(field,in_file_fie.c_str());

          FILE *fp_in_fie, *fp_in_fie_id;
          fp_in_fie = fopen(field, "rb");
          assert(fp_in_fie!=NULL);
          #ifdef SHIFT
              // input files(.field_ids): string >> char
          char* field_id = new char[in_file_fie_id.size() + 1];
          strcpy(field_id,in_file_fie_id.c_str());

          fp_in_fie_id = fopen(field_id, "rb");
          assert(fp_in_fie_id!=NULL);
          #endif

          // name of input files(.field): string >> char
          char* name_field = new char[in_file_name_fie.size() + 1];
          strcpy(name_field,in_file_name_fie.c_str());

          fprintf(stdout,"# Now, reading file: %s, No. %d\n",name_field, j);

          num_fie = B;
          while (num_fie == B) {
            // reading .field with BUFFERSIZE series into structure x_v[]
            num_fie = fread(x_v, sizeof(x_v[0]), B, fp_in_fie);
            #ifdef SHIFT
            num_fie_id = fread(ID, sizeof(ID[0]), B, fp_in_fie_id);
            assert(num_fie == num_fie_id);
            #endif
            vector<long int> k_v(num_fie);
            iota(k_v.begin(), k_v.end(), 0);
            // shuffle vector k_v
            shuffle(k_v.begin(), k_v.end(), mts);
            int j=0;
            long int k = 0;
            int z_off = 1;
            //#pragma omp parallel for MY_SCHEDULE
            for(int i=0; i<num_fie; i+=n_p)
            {
                // uniform random number in the rage of [0, num_fie-1]
                //uniform_int_distribution<> randf(0, (num_fie-1));
                //k = randf(mts);

              k = k_v[i];
                //moving and castig x_v[] and weight to x_w[]
              x_w[j].x[0] = (double)(x_v[k].x[0]*U_x) + (-0.5 - p_min_D)*L;
              x_w[j].x[1] = (double)(x_v[k].x[1]*U_x) + (-0.5 - p_min_D)*L;
                #ifdef index_rsd
              x_w[j].x[2] = (double)((x_v[k].x[2] + x_v[k].v[2]*U_v)*U_x) + (-0.5 - p_min_D)*L;
                #else
              x_w[j].x[2] = (double)(x_v[k].x[2]*U_x) + (-0.5 - p_min_D)*L;
                #endif
              x_w[j].w = weight;
                #ifdef periodic
              if(x_w[j].x[0] < p_min*L) x_w[j].x[0] += L;
              if(x_w[j].x[0] >= p_max*L) x_w[j].x[0] -= L;
              if(x_w[j].x[1] < p_min*L) x_w[j].x[1] += L;
              if(x_w[j].x[1] >= p_max*L) x_w[j].x[1] -= L;
              if(x_w[j].x[2] < p_min*L) x_w[j].x[2] += L;
              if(x_w[j].x[2] >= p_max*L) x_w[j].x[2] -= L;
                #endif
                #if defined SHIFT
                    //k = i;
              PIDt = ID[k].id;
              while(z_off){
                iz_ori = PIDt%ppd_int;
                iy_ori = ((PIDt - iz_ori)/ppd_int)%ppd_int;
                ix_ori = (((PIDt - iz_ori)/ppd_int) - iy_ori)/ppd_int;
                x_s0[j].x[0] = (double)(ix_ori)*L/ppd - L/2.0;
                x_s0[j].x[1] = (double)(iy_ori)*L/ppd - L/2.0;
                x_s0[j].x[2] = (double)(iz_ori)*L/ppd - L/2.0;
                assert(x_s0[j].x[0] >= - 0.5*L && x_s0[j].x[0] < 0.5*L);
                assert(x_s0[j].x[1] >= - 0.5*L && x_s0[j].x[1] < 0.5*L);
                assert(x_s0[j].x[2] >= - 0.5*L && x_s0[j].x[2] < 0.5*L);
                x_s1[j].x[0] = x_s0[j].x[0];
                x_s2[j].x[0] = x_s0[j].x[0];
                x_s1[j].x[1] = x_s0[j].x[1];
                x_s2[j].x[1] = x_s0[j].x[1];
                x_s1[j].x[2] = x_s0[j].x[2];
                x_s2[j].x[2] = x_s0[j].x[2];

                x_s0[j].w = (double)(x_v[k].x[0]*U_x) - x_s0[j].x[0];
                x_s1[j].w = (double)(x_v[k].x[1]*U_x) - x_s1[j].x[1];
                    #if defined(index_rsd)
                x_s2[j].w = (double)((x_v[k].x[2] + x_v[k].v[2]*U_v)*U_x) - x_s2[j].x[2];
                    #else
                x_s2[j].w = (double)(x_v[k].x[2]*U_x) - x_s2[j].x[2];
                    #endif
                if(x_s0[j].w < - 0.5*L) x_s0[j].w += L;
                if(x_s0[j].w >= 0.5*L) x_s0[j].w -= L;
                if(x_s1[j].w < - 0.5*L) x_s1[j].w += L;
                if(x_s1[j].w >= 0.5*L) x_s1[j].w -= L;
                if(x_s2[j].w < - 0.5*L) x_s2[j].w += L;
                if(x_s2[j].w >= 0.5*L) x_s2[j].w -= L;
                    /*
                    fprintf(stderr,"fie_tru:  ID = %lld, x_ini = %e, y_ini = %e, z_ini = %e\n",
                                PIDt, x_w[j].x[0], x_w[j].x[1], x_w[j].x[2]);
                    fprintf(stderr,"fie_tru:  x = %e, y = %e, z = %e, shift = %e\n",
                                x_v[k].x[0], x_v[k].x[1], x_v[k].x[2], x_w[j].w);
                    fprintf(stderr,"fie_tru:  v_x = %e, v_y = %e, v_z = %e\n",
                                x_v[k].v[0]*U_v, x_v[k].v[1]*U_v, x_v[k].v[2]*U_v);
                    */
                if(x_s2[j].w < -80.0){
                  PIDt -= 256;
                }
                else break;

              }
              PID.push_back(PIDt);
                #endif
              j++;
            }
            // outputting structure x_w[] to file_D
            fwrite(x_w, sizeof(x_w[0]), j, fp_out_D);
            #if defined SHIFT
            fwrite(x_s0, sizeof(x_s0[0]), j, fp_out_D_S0);
            fwrite(x_s1, sizeof(x_s1[0]), j, fp_out_D_S1);
            fwrite(x_s2, sizeof(x_s2[0]), j, fp_out_D_S2);
            #endif
            sum_num_fie += num_fie;
            sum_num_used_fie += j; //floor(num_fie/n_p);
            num_loop_f += 1;
          }
          fclose(fp_in_fie);
      #else  //HALO

        //******* .halos *********

          // input files(.halos): string >> char
          char* halos = new char[in_file_halo.size() + 1];
          strcpy(halos,in_file_halo.c_str());

          FILE *fp_in_halo, *fp_in_halo_id;
          fp_in_halo = fopen(halos, "rb");
          assert(fp_in_halo!=NULL);

          #ifdef SHIFT
            // input files(.halo_ids): string >> char
          char* halo_id = new char[in_file_halo_id.size() + 1];
          strcpy(halo_id,in_file_halo_id.c_str());

          fp_in_halo_id = fopen(halo_id, "rb");
          assert(fp_in_halo_id!=NULL);
          #endif

          // name of input files(.halos): string >> char
          char* name_halos = new char[in_file_name_halo.size() + 1];
          strcpy(name_halos,in_file_name_halo.c_str());

          fprintf(stdout,"# Now, reading file: %s, No. %d\n",name_halos, j);

          #ifdef HALO
            #if defined (planck) || defined (BOSS)
          uint64 num_groups[0], n_largest_subhalos[0], N_ave;
          num_halo = fread(num_groups, sizeof(num_groups[0]), 1, fp_in_halo);
          num_halo = fread(n_largest_subhalos, sizeof(n_largest_subhalos[0]), 1, fp_in_halo);
          fprintf(stderr,"# Number of halos: %lld\n", num_groups[0]);
          N_ave = 0;
            #endif
          #endif

          num_halo = B;
          while (num_halo == B) {
            // reading .halos with BUFFERSIZE series into structure x_v[]
            num_halo = fread(x_v, sizeof(x_v[0]), B, fp_in_halo);
            #ifdef SHIFT
            num_halo_id = fread(ID, sizeof(ID[0]), B, fp_in_halo_id);
            assert(num_halo == num_halo_id);
            #endif
            vector<long int> k_v(num_halo);
            iota(k_v.begin(), k_v.end(), 0);
            // shuffle vector k_v
            shuffle(k_v.begin(), k_v.end(), mts);
            int j = 0;
            long int k = 0;
            int z_off = 1;

            //#pragma omp parallel for MY_SCHEDULE
            for(int i=0; i<num_halo; i+=n_p)
            {
                // uniform random number in the rage of [0, num_halo-1)]
                //uniform_int_distribution<> randp(0, (num_halo-1));
                //k = randp(mts);

                //k = k_v[i];
              k = i;
                #ifdef HALO
              double cut_off_mass;
              cut_off_mass = pow(10.0,((double)M_cut));
                  #ifdef ST
                    //cutting off with mass
              if(x_v[k].m < cut_off_mass) continue;
                  #else
                    //counting the number of particles in each halo
              N_ave += x_v[k].N;
                    //cutting off with mass
              if(((double)MP)*x_v[k].N < cut_off_mass) continue;
                  #endif
                #endif
                #if defined(planck) && defined(GAL) // GRAND-HOD from Rockstar
                  //moving and castig x_v[] and weight to x_w[]
              x_w[j].x[0] = (double)(x_v[k].x[0]*U_x) - L*round((double)(x_v[k].x[0]*U_x)/L);
              x_w[j].x[1] = (double)(x_v[k].x[1]*U_x) - L*round((double)(x_v[k].x[1]*U_x)/L);
              x_w[j].x[2] = (double)(x_v[k].x[2]*U_x) - L*round((double)(x_v[k].x[2]*U_x)/L);
                #else
                  //moving and castig x_v[] and weight to x_w[]
              x_w[j].x[0] = (double)(x_v[k].x[0]*U_x) + (-0.5 - p_min_D)*L;
              x_w[j].x[1] = (double)(x_v[k].x[1]*U_x) + (-0.5 - p_min_D)*L;
                  #if defined (index_rsd)
              x_w[j].x[2] = (double)((x_v[k].x[2] + x_v[k].v[2]*U_v)*U_x) + (-0.5 - p_min_D)*L;
                  #else
              x_w[j].x[2] = (double)(x_v[k].x[2]*U_x) + (-0.5 - p_min_D)*L;
                  #endif
                #endif
              x_w[j].w = weight;
                #ifdef periodic
              if(x_w[j].x[0] < p_min*L) x_w[j].x[0] += L;
              if(x_w[j].x[0] >= p_max*L) x_w[j].x[0] -= L;
              if(x_w[j].x[1] < p_min*L) x_w[j].x[1] += L;
              if(x_w[j].x[1] >= p_max*L) x_w[j].x[1] -= L;
              if(x_w[j].x[2] < p_min*L) x_w[j].x[2] += L;
              if(x_w[j].x[2] >= p_max*L) x_w[j].x[2] -= L;
                #endif
              p_max_a = fmax(p_max_a,x_w[j].x[1]);
              p_min_a = fmin(p_min_a,x_w[j].x[1]);
                #if defined SHIFT
                    //k = i;
              PIDt = ID[k].id;
              while(z_off){
                iz_ori = PIDt%ppd_int;
                iy_ori = ((PIDt - iz_ori)/ppd_int)%ppd_int;
                ix_ori = (((PIDt - iz_ori)/ppd_int) - iy_ori)/ppd_int;
                x_s0[j].x[0] = (double)(ix_ori)*L/ppd - L/2.0;
                x_s0[j].x[1] = (double)(iy_ori)*L/ppd - L/2.0;
                x_s0[j].x[2] = (double)(iz_ori)*L/ppd - L/2.0;
                    //  fprintf(stderr,"= %e\n",x_s0[j].x[0]);
                assert(x_s0[j].x[0] >= - 0.5*L && x_s0[j].x[0] < 0.5*L);
                assert(x_s0[j].x[1] >= - 0.5*L && x_s0[j].x[1] < 0.5*L);
                assert(x_s0[j].x[2] >= - 0.5*L && x_s0[j].x[2] < 0.5*L);
                x_s1[j].x[0] = x_s0[j].x[0];
                x_s2[j].x[0] = x_s0[j].x[0];
                x_s1[j].x[1] = x_s0[j].x[1];
                x_s2[j].x[1] = x_s0[j].x[1];
                x_s1[j].x[2] = x_s0[j].x[2];
                x_s2[j].x[2] = x_s0[j].x[2];

                x_s0[j].w = (double)(x_v[k].x[0]*U_x) - x_s0[j].x[0];
                x_s1[j].w = (double)(x_v[k].x[1]*U_x) - x_s1[j].x[1];
                    #if defined(index_rsd)
                x_s2[j].w = (double)((x_v[k].x[2] + x_v[k].v[2]*U_v)*U_x) - x_s2[j].x[2];
                    #else
                x_s2[j].w = (double)(x_v[k].x[2]*U_x) - x_s2[j].x[2];
                    #endif
                if(x_s0[j].w < - 0.5*L) x_s0[j].w += L;
                if(x_s0[j].w >= 0.5*L) x_s0[j].w -= L;
                if(x_s1[j].w < - 0.5*L) x_s1[j].w += L;
                if(x_s1[j].w >= 0.5*L) x_s1[j].w -= L;
                if(x_s2[j].w < - 0.5*L) x_s2[j].w += L;
                if(x_s2[j].w >= 0.5*L) x_s2[j].w -= L;
                    /*
                    fprintf(stderr,"par_tru:  ID = %lld, x_ini = %e, y_ini = %e, z_ini = %e\n",
                                PIDt, x_w[j].x[0], x_w[j].x[1], x_w[j].x[2]);
                    fprintf(stderr,"par_tru:  x = %e, y = %e, z = %e, shift = %e\n",
                                x_v[k].x[0], x_v[k].x[1], x_v[k].x[2], x_w[j].w);
                    fprintf(stderr,"par_tru:  v_x = %e, v_y = %e, v_z = %e\n",
                                x_v[k].v[0]*U_v, x_v[k].v[1]*U_v, x_v[k].v[2]*U_v);
                    */
                if(x_s2[j].w < -80.0){
                  PIDt -= 256;
                }
                else break;
              }
              PID.push_back(PIDt);
                #endif
              if(x_w[j].x[0] > (p_min*L + C) ||
               x_w[j].x[1] > (p_min*L + C) ||
               x_w[j].x[2] > (p_min*L + C)){
               continue;
           }
           j++;
         }
            // outputting structure x_w[] to file_D
         fwrite(x_w, sizeof(x_w[0]), j, fp_out_D);
            #if defined SHIFT
         fwrite(x_s0, sizeof(x_s0[0]), j, fp_out_D_S0);
         fwrite(x_s1, sizeof(x_s1[0]), j, fp_out_D_S1);
         fwrite(x_s2, sizeof(x_s2[0]), j, fp_out_D_S2);
            #endif
         sum_num_halo += num_halo;
            sum_num_used_halo += j;//floor(num_halo/n_p);
            num_loop_p += 1;
          }
          #ifdef HALO
            #if defined (planck) || defined (BOSS)
          N_ave = N_ave/num_groups[0];
          fprintf(stderr,"# Averaged number of partiles in each halo: %lld\n", N_ave);
            #endif
          #endif
          fclose(fp_in_halo);
      #endif //HALO
    #else   //INITIAL
        //******* initial *********

          // input files(ic): string >> char
          char* ini = new char[in_file_ini.size() + 1];
          strcpy(ini,in_file_ini.c_str());

          // name of input files(ic): string >> char
          char* name_initial = new char[in_file_name_ini.size() + 1];
          strcpy(name_initial,in_file_name_ini.c_str());

          FILE *fp_in_ini;
          fp_in_ini = fopen(ini, "rb");
          assert(fp_in_ini!=NULL);
          fprintf(stdout,"# Now, reading file: %s, No. %d\n",name_initial, j);

          num_ini = B;
          while (num_ini == B) {
            // reading ic with BUFFERSIZE series into class x_ini[]
            num_ini = fread(x_ini, sizeof(x_ini[0]), B, fp_in_ini);
            //fprintf(stderr, "x = %d, y = %d, z = %d\n", x_ini[0].i,x_ini[0].j,x_ini[0].k);
            assert((ppd_int*ppd_int)%B == 0);
            sum_num_ini += num_ini;
            // choosing x axis
            if(x_ini[0].i%n_p_one != 0) continue;
            int j = 0;
            long int k = 0;
            //#pragma omp parallel for MY_SCHEDULE
            //for(int i=0; i<num_ini; i+=n_p_one)
            for(int i=0; i<num_ini; i+=1)
            {
                // choosing y and z axis
              if(x_ini[i].j%n_p_one == 0 && x_ini[i].k%n_p_one == 0) k = i;
              else continue;
                //moving and castig x_ini[] and weight to x_w[]
              double x_i, y_i, z_i;
              x_i = (p_min + x_ini[k].i/ppd)*L;
              y_i = (p_min + x_ini[k].j/ppd)*L;
              z_i = (p_min + x_ini[k].k/ppd)*L;
                //fprintf(stderr, "x = %d, y = %d, z = %d \n", x_ini[k].i, x_ini[k].j, x_ini[k].k);
              x_w[j].x[0] = (double)(x_i + x_ini[k].displ[0]);
              x_w[j].x[1] = (double)(y_i + x_ini[k].displ[1]);
                #ifdef index_rsd
              x_w[j].x[2] = (double)(z_i + x_ini[k].displ[2] + x_ini[k].vel[2]);
                #else
              x_w[j].x[2] = (double)(z_i + x_ini[k].displ[2]);
                #endif

                #ifdef periodic
              if(x_w[j].x[0] < p_min*L) x_w[j].x[0] += L;
              if(x_w[j].x[0] >= p_max*L) x_w[j].x[0] -= L;
              if(x_w[j].x[1] < p_min*L) x_w[j].x[1] += L;
              if(x_w[j].x[1] >= p_max*L) x_w[j].x[1] -= L;
              if(x_w[j].x[2] < p_min*L) x_w[j].x[2] += L;
              if(x_w[j].x[2] >= p_max*L) x_w[j].x[2] -= L;
                #endif

              x_w[j].w = weight;
              j++;
            }
            // outputting structure x_w[] to file_D
            fwrite(x_w, sizeof(x_w[0]), j, fp_out_D);
            sum_num_used_ini += j;//floor(num_ini/n_p);
            num_loop_i += 1;
          }
          fclose(fp_in_ini);
    #endif  //INITIAL
          fclose(fp_out_D);
    #ifdef SHIFT
          fclose(fp_out_D_S0);
          fclose(fp_out_D_S1);
          fclose(fp_out_D_S2);
    #endif

          sum_num = sum_num_par + sum_num_fie + sum_num_halo + sum_num_ini;
          sum_num_used = sum_num_used_par + sum_num_used_fie + sum_num_used_halo + sum_num_used_ini;

      // computing the sum of the number of particles, field, and the both for each file (halos_[x])
          fprintf(stdout,"# num_par = %ld, num_fie = %ld, num_halo = %ld, num_ini = %ld, sum = %ld\n"
            ,sum_num_par, sum_num_fie, sum_num_halo, sum_num_ini, sum_num);

          tot_par += sum_num_par;
          tot_fie += sum_num_fie;
          tot_halo += sum_num_halo;
          tot_ini += sum_num_ini;
          tot_D += sum_num;
          used_D += sum_num_used;

          sum_num_par = 0;
          sum_num_fie = 0;
          sum_num_halo = 0;
          sum_num_ini = 0;
          sum_num = 0;
          sum_num_used_par = 0;
          sum_num_used_fie = 0;
          sum_num_used_halo = 0;
          sum_num_used_ini = 0;
          sum_num_used = 0;

        }
        fprintf(stderr,"# max and min of position: %f and %f\n", p_max_a, p_min_a);

    // computing the total number of particles, field, and the both
        fprintf(stdout,"\n# tot_par = %ld, tot_fie = %ld, tot_halo = %ld, tot_ini = %ld, tot_D = %ld, used_D = %ld\n\n"
          ,tot_par, tot_fie, tot_halo, tot_ini, tot_D, used_D);
        fprintf(stderr,"# tot_par = %ld, tot_fie = %ld, tot_halo = %ld, tot_ini = %ld, tot_D = %ld, used_D = %ld\n\n"
          ,tot_par, tot_fie, tot_halo, tot_ini, tot_D, used_D);
        Data_t.Stop();
//#endif // SHIFT fake

    #if defined SHIFT && defined initial
      //******* Data_ini *********

        Data_ini_t.Start();

        long long int p_i = 0;

      /*
      // creating a vector with the same number of particles as PID
      vec_set_t.Start();
      #pragma omp parallel for MY_SCHEDULE
      for(uint64 i=0; i<PID.size(); i++){
        PID[i] = pow(3200, 3)/PID.size()*i;
      }
      vec_set_t.Stop();
      */

      // std::sort
        vec_set_t.Start();
        sort(PID.begin(), PID.end());
        vec_set_t.Stop();


      /*
      // merge sort
      vec_set_t.Start();
      PID = mergeSort(PID);
      vec_set_t.Stop();
      */

        fprintf(stderr,"\n# PID set = %e\n", vec_set_t.Elapsed());
        fprintf(stderr,"# = %lld, = %lld\n\n", *PID.begin(), *(PID.end()-1));


      // itarating "num_files" times for all files halos_[x]
        int num_files_2, cur_num_file;
        num_files_2 = file_number_ini;
        for(int j=0; j<num_files_2; j++){
          cur_num_file = j;

        // j: int >> string
          stringstream ss;
          ss << j;
          string num_file = ss.str();

        // initial
          string in_file_ini = in_path_ini + "ic_" + num_file;
          string in_file_name_ini = "ic_" + num_file;

          FILE *fp_out_D_ini;
          if(j == 0){
          fp_out_D_ini = fopen(out_D_ini, "wb");  // from the beginning（j＝0）
          assert(fp_out_D_ini!=NULL);
          // output: header
          fwrite(&header, sizeof(header), 1, fp_out_D_ini);
        }
        else{
          fp_out_D_ini = fopen(out_D_ini, "ab");  // from the middle (j>=1)
          assert(fp_out_D_ini!=NULL);
        }

            // input files(ic): string >> char
        char* ini = new char[in_file_ini.size() + 1];
        strcpy(ini,in_file_ini.c_str());

            // name of input files(ic): string >> char
        char* name_initial = new char[in_file_name_ini.size() + 1];
        strcpy(name_initial,in_file_name_ini.c_str());

        FILE *fp_in_ini;
        fp_in_ini = fopen(ini, "rb");
        assert(fp_in_ini!=NULL);
        fprintf(stderr,"# Now, reading file: %s, No. %d\n",name_initial, j);
        fprintf(stdout,"# Now, reading file: %s, No. %d\n",name_initial, j);
        long int num_infile = 0;
        num_ini = B;
        while (num_ini == B) {
              // reading ic with BUFFERSIZE series into class x_ini[]
          num_ini = fread(x_ini, sizeof(x_ini[0]), B, fp_in_ini);
          if(num_ini == 0) break;
          long int k = 0;
          long int k_max = 0;
          long int k_arr[B];

          for(long int j = 0; j < B; j++){
            num_blo = PID[p_i+j]/1310720000;
            num_sub_blo = (PID[p_i+j]%1310720000)/112640000;
            num_res = (PID[p_i+j]%1310720000)%112640000;
            num_file_ini_tem = num_blo*35 + num_sub_blo*3;

            if(num_sub_blo != 11){
              if(num_res >= 81920000) {
                num_res -= 81920000;
                num_file_ini_tem += 2;
              }
              else if(num_res >= 40960000){
                num_res -= 40960000;
                num_file_ini_tem += 1;
              }
            }
            else{
              if(num_res >= 40960000){
               num_res -= 40960000;
               num_file_ini_tem += 1;
             }
           }
           k_arr[j] = num_res - num_infile*B;
           if(k_arr[j] >= B || k_arr[j] < 0.0) break;
           k_max ++;
         }

              //#pragma omp parallel for MY_SCHEDULE
         for(long int j = 0; j < k_max; j++)
         {
          k = k_arr[j];
          iz_ori = x_ini[k].k;
          iy_ori = x_ini[k].j;
          ix_ori = x_ini[k].i;

          x_w[j].x[0] = (double)(ix_ori)*L/ppd - L/2.0;
          x_w[j].x[1] = (double)(iy_ori)*L/ppd - L/2.0;
          x_w[j].x[2] = (double)(iz_ori)*L/ppd - L/2.0;
          assert(x_w[j].x[0] >= - 0.5*L && x_w[j].x[0] < 0.5*L);
          assert(x_w[j].x[1] >= - 0.5*L && x_w[j].x[1] < 0.5*L);
          assert(x_w[j].x[2] >= - 0.5*L && x_w[j].x[2] < 0.5*L);
                  #if defined(index_rsd) && (AXIS == 2)
          x_w[j].w = (double)(x_ini[k].displ[axis] + x_ini[k].vel[axis]);
                  #else
          x_w[j].w = (double)x_ini[k].displ[axis];
                  #endif
          if(x_w[j].w < - 0.5*L) x_w[j].w += L;
          if(x_w[j].w >= 0.5*L) x_w[j].w -= L;
                  /*
                  fprintf(stderr,"par_ini:  ID = %lld, x_ini = %e, y_ini = %e, z_ini = %e, shift = %e\n",
                              PID[p_i+j], x_w[j].x[0], x_w[j].x[1], x_w[j].x[2], x_w[j].w);
                  fprintf(stderr,"par_ini:  v_x = %e, v_y = %e, v_z = %e\n\n",
                              x_ini[k].vel[0], x_ini[k].vel[1], x_ini[k].vel[2]);
                  */
        }
              // outputting structure x_w[] to file_D
        fwrite(x_w, sizeof(x_w[0]), k_max, fp_out_D_ini);
        sum_num_ini += num_ini;
              sum_num_used_ini += k_max;  //floor(num_ini/n_p);
              num_loop_i += 1;
              p_i += k_max;
              num_infile ++;
            }
            fclose(fp_in_ini);
            fclose(fp_out_D_ini);

            sum_num = sum_num_ini;
            sum_num_used = sum_num_used_ini;

        // computing the sum of the number of particles, field, and the both for each file (halos_[x])
            fprintf(stdout,"# num_ini = %ld, sum = %ld\n",sum_num_ini, sum_num);

            tot_ini += sum_num_ini;
            tot_D += sum_num;
            used_D += sum_num_used;

            sum_num_ini = 0;
            sum_num = 0;
            sum_num_used_ini = 0;
            sum_num_used = 0;

          }
      // computing the total number of particles, field, and the both
          fprintf(stdout,"\n# tot_par = %ld, tot_fie = %ld, tot_ini = %ld, tot_D = %ld, used_D = %ld\n"
            ,tot_par, tot_fie, tot_ini, tot_D, used_D);
          fprintf(stderr,"# tot_par = %ld, tot_fie = %ld, tot_ini = %ld, tot_D = %ld, used_D = %ld\n"
            ,tot_par, tot_fie, tot_ini, tot_D, used_D);
          Data_ini_t.Stop();
    #endif
  #endif //only_R

  #ifdef only_R
          used_D = (long int)USED_Data;
  #endif

  //******* Random (>>filename2)*********

  #ifdef Random
          Random_t.Start();
          FILE *fp_out_R;
          fp_out_R = fopen(out_R, "wb");
          assert(fp_out_R!=NULL);

      // output: header
          fwrite(&header, sizeof(header), 1, fp_out_R);

      // non-deterministic random number generator
          random_device rnd;
      // Mersenne Twister, 64bit version
          mt19937_64 mt(rnd());
      // uniform random number in the rage of [-0.5, 0.5]
          uniform_real_distribution<> rand1(p_min, p_max);

      // total number of random
          tot_R_p = (long int)ra*used_D;

          tot_R = 0;

    #ifndef Uniform
      // just Random
          num_ran = BUFFERSIZE;
      // definition for "real" alpha
          alpha = used_D/tot_R_p;
      // output: definition for structure
          struct_out_x3w x_w_R[BUFFERSIZE];
          while (tot_R != tot_R_p) {
        //#pragma omp parallel for MY_SCHEDULE
            for(int k=0; k<num_ran; k++){
          //copying each position and weight to x_w_R[]
              x_w_R[k].x[0] = (double)(rand1(mt)*L);
              x_w_R[k].x[1] = (double)(rand1(mt)*L);
              x_w_R[k].x[2] = (double)(rand1(mt)*L);
              x_w_R[k].w = - alpha*weight;

              num_loop_R += 1;
            }
        // outputting structure x_w[] to file_R
            fwrite(x_w_R, sizeof(x_w_R[0]), num_ran, fp_out_R);
            tot_R += num_loop_R;
            num_loop_R = 0;

            if((tot_R_p - tot_R) >= BUFFERSIZE){
              num_ran = BUFFERSIZE;
            }
            else{
              num_ran = (tot_R_p - tot_R);
            }
          }
    #else //Uniform
      // "uniform" Random
          if(index_ratio_RtoD == -1){
            num_ran = ngrid_1d;
          }
          else{
            num_ran = ceil(pow(tot_R_p, 1.0/3.0));
          }
      // definition for "real" alpha
          alpha = used_D/pow(num_ran,3);
      // output: definition for structure
          struct_out_x3w x_w_R[num_ran];
        //#pragma omp parallel for MY_SCHEDULE
          for(int k=0; k<num_ran; k++){
            for(int j=0; j<num_ran; j++){
              for(int i=0; i<num_ran; i++){
              //copying each position and weight to x_w_R[]
                double x_g_p, y_g_p, z_g_p;
                x_g_p = p_min*L + C/num_ran*(i);
                y_g_p = p_min*L + C/num_ran*(j);
                z_g_p = p_min*L + C/num_ran*(k);
                x_w_R[i].x[0] = x_g_p;
                x_w_R[i].x[1] = y_g_p;
                x_w_R[i].x[2] = z_g_p;
              #ifdef periodic
                if(x_w_R[i].x[0] < p_min*L) x_w_R[i].x[0] += L;
                if(x_w_R[i].x[0] >= (p_min*L + C)) x_w_R[i].x[0] -= L;
                if(x_w_R[i].x[1] < p_min*L) x_w_R[i].x[1] += L;
                if(x_w_R[i].x[1] >= (p_min*L + C)) x_w_R[i].x[1] -= L;
                if(x_w_R[i].x[2] < p_min*L) x_w_R[i].x[2] += L;
                if(x_w_R[i].x[2] >= (p_min*L + C)) x_w_R[i].x[2] -= L;
              #endif
                x_w_R[i].w = - alpha*weight;
                num_loop_R += 1;
              }
            // outputting structure x_w[] to file_R
              fwrite(x_w_R, sizeof(x_w_R[0]), num_ran, fp_out_R);
              tot_R += num_loop_R;
              num_loop_R = 0;
            }
          }
    #endif //Uniform

      // computing the total number of randoms and alpha
          fprintf(stdout,"# tot_R = %ld, alpha = %e\n",tot_R, alpha);

          fclose(fp_out_R);
          Random_t.Stop();
  #endif //Random

          fprintf(stdout, "#   Data time:   %8.4f s\n", Data_t.Elapsed());
          fprintf(stdout, "#   Data_ini time:   %8.4f s\n", Data_ini_t.Elapsed());
          fprintf(stdout, "#   Random time:   %8.4f s\n", Random_t.Elapsed());
          fprintf(stdout, "\n");

  //fprintf(stderr, "Check\n");
#endif //CHECK




//******* check for output files *********

          FILE *fp_in;

    // twice for Data and Random
          for(int q=0; q<5; q++){
            if(q == 0){
        // for Data file
              fp_in = fopen(out_D, "rb");
              if(fp_in == NULL) continue;
              fprintf(stdout,"# Data (file_D)\n");
            }
      #if defined SHIFT
            else if(q == 1){
          // for Data file
              fp_in = fopen(out_D_S0, "rb");
              if(fp_in == NULL) continue;
              fprintf(stdout,"# Data (file_S0))\n");
            }
            else if(q == 2){
          // for Data file
              fp_in = fopen(out_D_S1, "rb");
              if(fp_in == NULL) continue;
              fprintf(stdout,"# Data (file_S1))\n");
            }
            else if(q == 3){
          // for Data file
              fp_in = fopen(out_D_S2, "rb");
              if(fp_in == NULL) continue;
              fprintf(stdout,"# Data (file_S2))\n");
            }
      #endif
            else{
        //for Random (or shift_ini) file
        #ifndef SHIFT
              fp_in = fopen(out_R, "rb");
              if(fp_in == NULL) continue;
              fprintf(stdout,"# Random (file_R)\n");
        #else
              fp_in = fopen(out_D_ini, "rb");
              if(fp_in == NULL) continue;
              fprintf(stdout,"# Data (file_S_ini)\n");
        #endif
            }

            int num;
            num = (int)CHECK_num;
            struct_out_x3w in_x_w1[num];
            struct_out_x3w in_x_w2[num];
            struct_header in_header;
            double in_cont[10];

            fread(&in_header, sizeof(in_header), 1, fp_in);

      //fseek(fp_in, (sizeof(in_header) + sizeof(in_x_w1[0])*0 + sizeof(double)*0), SEEK_SET);
      //fread(in_cont, sizeof(in_cont[0]), 10, fp_in);

            fseek(fp_in, (sizeof(in_header) + sizeof(in_x_w1[0])*(0)), SEEK_SET);
            fread(in_x_w1, sizeof(in_x_w1[0]), num, fp_in);

            fseek(fp_in, sizeof(in_x_w2[0])*(-num), SEEK_END);
            fread(in_x_w2, sizeof(in_x_w2[0]), num, fp_in);

            fclose(fp_in);

      //fprintf(stdout,"= %e, = %e,\n = %e, = %e\n",in_cont[0], in_cont[1], in_cont[2], in_cont[3]);

            fprintf(stdout,"Header: posmin_x = %e, posimax_x = %e, maxcep = %e, blank8 = %e\n",
              in_header.posmin[0],in_header.posmax[0],in_header.maxcep,in_header.blank8);
            for(int i=0; i < num; i++){
              fprintf(stdout,"No.%d from First: x = %e, y = %e, z = %e, w = %e\n",i,
                in_x_w1[i].x[0],in_x_w1[i].x[1],in_x_w1[i].x[2],in_x_w1[i].w);
            }
            for(int i=0; i < num; i++){
              fprintf(stdout,"No.%d to Last: x = %e, y = %e, z = %e, w = %e\n",i,
                in_x_w2[i].x[0],in_x_w2[i].x[1],in_x_w2[i].x[2],in_x_w2[i].w);
            }
            fprintf(stdout,"#####################\n\n\n\n");
          }

          return 0;
        }

//*********  merge sort *************//  http://www.bogotobogo.com/Algorithms/mergesort.php

  //  merge
        vector<uint64> merge(vector<uint64> left, vector<uint64> right)
        {
         vector<uint64> result;
         while ((uint64)left.size() > 0 || (uint64)right.size() > 0) {
          if ((uint64)left.size() > 0 && (uint64)right.size() > 0) {
           if ((uint64)left.front() <= (uint64)right.front()) {
            result.push_back((uint64)left.front());
            left.erase(left.begin());
          }
          else {
            result.push_back((uint64)right.front());
            right.erase(right.begin());
          }
        }  else if ((uint64)left.size() > 0) {
          for (uint64 i = 0; i < (uint64)left.size(); i++)
           result.push_back(left[i]);
         break;
       }  else if ((uint64)right.size() > 0) {
        for (uint64 i = 0; i < (uint64)right.size(); i++)
         result.push_back(right[i]);
       break;
     }
   }
   return result;
 }

  //  merge sort
 vector<uint64> mergeSort(vector<uint64> m)
 {
   if (m.size() <= 1)
    return m;

  vector<uint64> left, right, result;
  uint64 middle = ((uint64)m.size()+ 1) / 2;

  for (uint64 i = 0; i < middle; i++) {
    left.push_back(m[i]);
  }

  for (uint64 i = middle; i < (uint64)m.size(); i++) {
    right.push_back(m[i]);
  }

  left = mergeSort(left);
  right = mergeSort(right);
  result = merge(left, right);

  return result;
}
