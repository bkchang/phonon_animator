#include <thrust/complex.h>
#include <cuda_runtime.h>
#include "materialsystem.h"
#include "helper_cuda.h"

#pragma once

#define BIAS 1e-4
#define MAX_RAY_DEPTH 5
#define NUMBER_SPHERE_SPECIES 3
#define LEN_SPHERE_PROP_ARR 9
#define IOR 1.1

struct ray{
    float3 orig;
    float3 sc;
    float3 dir;
    float3 phit;
    float3 nhit;
    int hit_sphereID;
    bool inside;
};

struct sphere{
    float r;
    float3 sc;
    float refl;
    float transp;
    float3 ec;
};

extern sphere species_sphere_prop[];

__host__ __device__
unsigned int TwoToThePowOf (unsigned int p);

__global__
void cuda_get_sphere_pos_kernel(const unsigned int natom,
                                const float3 *sphere_pos_dev,
                                const float amplitude,
                                const thrust::complex<float> phase,
                                const thrust::complex<float> *eigvec_dev,
                                float3 *sphere_displaced_pos_dev);

__global__
void cuda_initialize_firstLevel_rays_dir_kernel (
                            const unsigned int width, const unsigned int height,
                            const float invWidth, const float invHeight,
                            const float angle, const float aspectratio,
                            ray* level_rays_dev);

__device__
bool cuda_intersect (const ray &r,
                const unsigned int sphereID,
                const sphere *species_sphere_prop_dev,
                const int *sphere_type_dev,
                const float3 *sphere_displaced_pos_dev,
                float &t0, float &t1);

__global__
void cuda_find_intersecting_sphere_kernel (const unsigned int width,
                                           const unsigned int height,
                                           const unsigned int nspheres,
                                           const sphere *species_sphere_prop_dev,
                                           const int *sphere_type_dev,
                                           const float3 *sphere_displaced_pos_dev,
                                           const unsigned int level,
                                           ray* level_rays_dev);

__global__
void cuda_trace_reflection_and_refraction_kernel (
                        const unsigned int width,
                        const unsigned int height,
                        const unsigned int level,
                        ray* this_level_rays_dev,
                        ray* next_level_rays_dev);

__device__
float cuda_mix(const float &a, const float &b, const float &mix);

__global__
void cuda_compute_last_level_Color_kernel(const unsigned int width,
                                          const unsigned int height,
                                          const unsigned int nspheres,
                                          const sphere *species_sphere_prop_dev,
                                          const int *sphere_type_dev,
                                          const float3 *sphere_displaced_pos_dev,
                                          ray* level_rays_dev);

__global__
void cuda_compute_Color_kernel(const unsigned int width,
                               const unsigned int height,
                               const unsigned int nspheres,
                               const sphere *species_sphere_prop_dev,
                               const int *sphere_type_dev,
                               const unsigned int level,
                               ray* this_level_rays_dev,
                               ray* next_level_rays_dev);

__global__
void cuda_collect_pixels(const unsigned int width,
                         const unsigned int height,
                         ray* rays_dev,
                         char* video_dev);

__host__
void renderVideo_gpu(const MaterialSystem matSys,
                 const unsigned imode = 1,
                 const unsigned width = 640,
                 const unsigned height = 480,
                 const unsigned nstep_per_cycle = 20,
                 const unsigned nloop = 5,
                 const float    amplitude = 0.2);