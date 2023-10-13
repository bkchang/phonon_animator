/**************************************
 * CUDA kernels and helper functions
 **************************************/
#include <cuda_runtime.h>
#include <thrust/complex.h>
#include <math.h>
#include <array>
#include "materialsystem.h"
#include "raytracer.cuh"
#include "helper_cuda.h"
#include "helper_math.h"

// SPHERE_H, SPHERE_C, SPHERE_LIGHT
sphere species_sphere_prop[3] = {
    {0.05, make_float3(0.90, 0.76, 0.46), 0.10, 0.40, make_float3(0.00, 0.00, 0.00)},
    {0.09, make_float3(0.65, 0.77, 0.97), 0.10, 0.40, make_float3(0.00, 0.00, 0.00)},
    {3.00, make_float3(0.00, 0.00, 0.00), 0.00, 0.00, make_float3(3.00, 3.00, 3.00)}};

__host__ __device__
unsigned int TwoToThePowOf (unsigned int p) {
    unsigned int x = 1;
    for (unsigned int i = 0; i < p; i++)
        x *= 2;
    return x;
}

__global__
void cuda_get_sphere_pos_kernel(const unsigned int natom,
                                const float3 *sphere_pos_dev,
                                const float amplitude,
                                const thrust::complex<float> phase,
                                const thrust::complex<float> *eigvec_dev,
                                float3 *sphere_displaced_pos_dev) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    while (i < natom) {
        sphere_displaced_pos_dev[i].x = sphere_pos_dev[i].x + amplitude * (phase * eigvec_dev[i*3+0]).real();
        sphere_displaced_pos_dev[i].y = sphere_pos_dev[i].y + amplitude * (phase * eigvec_dev[i*3+1]).real();
        sphere_displaced_pos_dev[i].z = sphere_pos_dev[i].z + amplitude * (phase * eigvec_dev[i*3+2]).real();
        i += blockDim.x * gridDim.x;
    }
}

__global__
void cuda_initialize_firstLevel_rays_dir_kernel (
                            const unsigned int width, const unsigned int height,
                            const float invWidth, const float invHeight,
                            const float angle, const float aspectratio,
                            ray* level_rays_dev) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned x, y;
    while (i < width * height) {
        y = i / width;
        x = i - y * width;
        level_rays_dev[i].dir.x = (2 * ((x + 0.5) * invWidth) - 1) * angle * aspectratio;
        level_rays_dev[i].dir.y = (1 - 2 * ((y + 0.5) * invHeight)) * angle;
        level_rays_dev[i].dir.z = -1.0;
        level_rays_dev[i].dir = normalize(level_rays_dev[i].dir);
        i += blockDim.x * gridDim.x;
    }
}

__device__
bool cuda_intersect (ray &r,
                     const unsigned int sphereID,
                     const sphere* species_sphere_prop_dev,
                     const int *sphere_type_dev,
                     const float3 *sphere_displaced_pos_dev,
                     float &t0, float &t1) {
    float3 l = sphere_displaced_pos_dev[sphereID] - r.orig;
    float tca = dot(l, r.dir);
    if (tca < 0) return false;
    float d2 = dot(l, l) - tca * tca;
    float radius = species_sphere_prop_dev[sphere_type_dev[sphereID]].r;
    float radius2 = radius * radius;
    if (d2 > radius2) return false;
    float thc = sqrtf (radius2 - d2);
    t0 = tca - thc;
    t1 = tca + thc;
    return true;
}

__global__
void cuda_find_intersecting_sphere_kernel (const unsigned int width,
                                           const unsigned int height,
                                           const unsigned int nspheres,
                                           const sphere *species_sphere_prop_dev,
                                           const int *sphere_type_dev,
                                           const float3 *sphere_displaced_pos_dev,
                                           const unsigned int level,
                                           ray* level_rays_dev) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    float tnear, t0, t1;
    while (i < TwoToThePowOf(level) * width * height) {
        tnear = INFINITY;
        level_rays_dev[i].hit_sphereID = -1;
        for (unsigned j = 0; j < nspheres; j++) {
            t0 = INFINITY, t1 = INFINITY;
            if (cuda_intersect(level_rays_dev[i], j, \
                          species_sphere_prop_dev, sphere_type_dev, sphere_displaced_pos_dev, \
                          t0, t1)) {
                if (t0 < 0) t0 = t1;
                if (t0 < tnear) {
                    tnear = t0;
                    level_rays_dev[i].hit_sphereID = j;
                }
            }
            
        }
        // Below is relevant only if level_rays_dev[i].hit_sphereID != -1
        if (level_rays_dev[i].hit_sphereID != -1) {
            level_rays_dev[i].phit = level_rays_dev[i].orig + level_rays_dev[i].dir * tnear;
            level_rays_dev[i].nhit = level_rays_dev[i].phit - sphere_displaced_pos_dev[level_rays_dev[i].hit_sphereID];
            level_rays_dev[i].nhit = normalize(level_rays_dev[i].nhit);
            level_rays_dev[i].inside = false;
            if (dot(level_rays_dev[i].dir, level_rays_dev[i].nhit) > 0) {
                level_rays_dev[i].nhit = -level_rays_dev[i].nhit;
                level_rays_dev[i].inside = true;
            }
        }
        i += blockDim.x * gridDim.x;
    }    
}

__global__
void cuda_trace_reflection_and_refraction_kernel (
                        const unsigned int width,
                        const unsigned int height,
                        const unsigned int level,
                        ray* this_level_rays_dev,
                        ray* next_level_rays_dev) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    float eta, cosi, k;
    while (i < TwoToThePowOf(level) * width * height) {
        // reflection
        next_level_rays_dev[2 * i].orig = this_level_rays_dev[i].phit + \
                                      this_level_rays_dev[i].nhit * BIAS;
        next_level_rays_dev[2 * i].dir  = normalize( this_level_rays_dev[i].dir - \
                                      this_level_rays_dev[i].nhit * 2 * \
                                      dot(this_level_rays_dev[i].dir, this_level_rays_dev[i].nhit) );
        // refraction
        eta = (this_level_rays_dev[i].inside) ? IOR : 1 / IOR;
        cosi = -dot(this_level_rays_dev[i].nhit, this_level_rays_dev[i].dir);
        k = 1 - eta * eta * (1 - cosi * cosi);
        next_level_rays_dev[2 * i + 1].orig = \
                                      this_level_rays_dev[i].phit - \
                                      this_level_rays_dev[i].nhit * BIAS;
        next_level_rays_dev[2 * i + 1].dir = \
                                      normalize( this_level_rays_dev[i].dir * 
                                      eta + this_level_rays_dev[i].nhit * (eta
                                      * cosi - sqrtf(k)) ); \    

        i += blockDim.x * gridDim.x;
    }
}

__device__
float cuda_mix(const float &a, const float &b, const float &mix) {
    return b * mix + a * (1 - mix);
}

__global__
void cuda_compute_last_level_Color_kernel(const unsigned int width,
                                          const unsigned int height,
                                          const unsigned int nspheres,
                                          const sphere *species_sphere_prop_dev,
                                          const int *sphere_type_dev,
                                          const float3 *sphere_displaced_pos_dev,
                                          ray* level_rays_dev) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    float3 sphereSurfaceColor, sphereiEmissionColor;
    float3 transmission;
    float t0, t1;
    ray r;
    while (i < TwoToThePowOf(MAX_RAY_DEPTH) * width * height) {
        sphereSurfaceColor = species_sphere_prop_dev[sphere_type_dev[level_rays_dev[i].hit_sphereID]].sc;                      
        for (unsigned int is = 0; is < nspheres; is++) {
            if (species_sphere_prop_dev[sphere_type_dev[is]].ec.x > 0) {
                sphereiEmissionColor = species_sphere_prop_dev[sphere_type_dev[is]].ec;
                transmission = make_float3(1);
                r.dir = normalize(sphere_displaced_pos_dev[is] - level_rays_dev[i].phit); // lightDirection
                r.orig = level_rays_dev[i].phit + level_rays_dev[i].nhit * BIAS;
                for (unsigned int js = 0; js < nspheres; js++) {
                    if (js != is) {
                        if (cuda_intersect(r, js, species_sphere_prop_dev, sphere_type_dev, sphere_displaced_pos_dev, t0, t1)) {
                            transmission = make_float3(0);
                            break;
                        }
                    }
                }
                level_rays_dev[i].sc += sphereSurfaceColor * transmission * \
                                        fmaxf(0.0, dot(level_rays_dev[i].nhit, r.dir)) * sphereiEmissionColor;
            }
        }
        level_rays_dev[i].sc += species_sphere_prop_dev[sphere_type_dev[level_rays_dev[i].hit_sphereID]].ec;
        i += blockDim.x * gridDim.x;
    }
}

__global__
void cuda_compute_Color_kernel(const unsigned int width,
                               const unsigned int height,
                               const unsigned int nspheres,
                               const sphere *species_sphere_prop_dev,
                               const int *sphere_type_dev,
                               const unsigned int level,
                               ray* this_level_rays_dev,
                               ray* next_level_rays_dev) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    float facingratio, fresneleffect;
    while (i < TwoToThePowOf(level) * width * height) {
        if (this_level_rays_dev[i].hit_sphereID == -1) {
            this_level_rays_dev[i].sc = make_float3(2);
        } else {
            // from reflection
            facingratio = -dot(this_level_rays_dev[i].dir, this_level_rays_dev[i].nhit);
            fresneleffect = cuda_mix(powf(1 - facingratio, 3.0), 1.0, 0.1);
            this_level_rays_dev[i].sc = (next_level_rays_dev[2 * i].sc * fresneleffect + \
                                         next_level_rays_dev[2 * i + 1].sc * (1 - fresneleffect) * 
                                         species_sphere_prop_dev[sphere_type_dev[this_level_rays_dev[i].hit_sphereID]].transp) * \
                                         species_sphere_prop_dev[sphere_type_dev[this_level_rays_dev[i].hit_sphereID]].sc;
            this_level_rays_dev[i].sc += species_sphere_prop_dev[sphere_type_dev[this_level_rays_dev[i].hit_sphereID]].ec;
        }
        i += blockDim.x * gridDim.x;
    }
}

__global__
void cuda_collect_pixels(const unsigned int width,
                         const unsigned int height,
                         ray* rays_dev,
                         char* video_dev) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    while (i < width * height) {
        video_dev[i] = (unsigned char)(fminf(float(1), rays_dev[i].sc.y) * 255);
        video_dev[i + width * height] = (unsigned char)(fminf(float(1), rays_dev[i].sc.z) * 255);
        video_dev[i + 2 * width * height] = (unsigned char)(fminf(float(1), rays_dev[i].sc.x) * 255);
        i += blockDim.x * gridDim.x;
    }
}