/**************************************
 * Main driver program for the GPU part
 **************************************/
#include <fstream>
#include <cuda_runtime.h>
#include <thrust/complex.h>
#include <array>
#include "materialsystem.h"
#include "raytracer.cuh"
#include "helper_cuda.h"
#include "helper_math.h"

__host__
void renderVideo_gpu(const MaterialSystem matSys,
                 const unsigned imode,
                 const unsigned width,
                 const unsigned height,
                 const unsigned nstep_per_cycle,
                 const unsigned nloop,
                 const float    amplitude)
{
    float invWidth = 1 / float(width), invHeight = 1 / float(height);
    float fov = 30, aspectratio = width / float(height);
    float angle = tan(M_PI * 0.5 * fov / 180.);
    std::ofstream tmpFile("./results/tmp.bin", std::ios::binary);

    /**************************************************************
     * The following variables are unchanged during the simulation,
     * so are saved to the GPU in the very beginning.
     *************************************************************/

    // Number of atoms
    unsigned int natom = matSys.natom;

    // Sphere property data
    sphere* species_sphere_prop_dev;
    CUDA_CALL( cudaMalloc((void**) &species_sphere_prop_dev, 3 * sizeof(sphere)) );
    CUDA_CALL( cudaMemcpy(species_sphere_prop_dev, species_sphere_prop, 3 * sizeof(sphere), cudaMemcpyHostToDevice) );

    // Sphere types (including the light source)
    int sphere_type[natom + 1];
    for (unsigned ia = 0; ia < natom; ia++) {
        if (matSys.atoms[ia].species == "H") {
            sphere_type[ia] = 0;
        } else if (matSys.atoms[ia].species == "C") {
            sphere_type[ia] = 1;
        }
    }
    sphere_type[natom] = 2;
    int* sphere_type_dev;
    CUDA_CALL( cudaMalloc((void**) &sphere_type_dev, (natom + 1) * sizeof(int)) );
    CUDA_CALL( cudaMemcpy(sphere_type_dev, sphere_type, (natom + 1) * sizeof(int), cudaMemcpyHostToDevice) );

    // Sphere original positions (including the light source)
    float3 sphere_pos[natom + 1];
    for (unsigned ia = 0; ia < natom; ia++) {
        sphere_pos[ia].x = matSys.atoms[ia].pos_cart_alat[0];
        sphere_pos[ia].y = matSys.atoms[ia].pos_cart_alat[1];
        sphere_pos[ia].z = matSys.atoms[ia].pos_cart_alat[2];
    }
    sphere_pos[natom].x =   0.0;
    sphere_pos[natom].y = -20.0;
    sphere_pos[natom].z = -30.0;
    float3* sphere_pos_dev;
    CUDA_CALL( cudaMalloc((void**) &sphere_pos_dev, (natom+1) * sizeof(float3)) );
    CUDA_CALL( cudaMemcpy(sphere_pos_dev, sphere_pos, (natom+1) * sizeof(float3), cudaMemcpyHostToDevice) );

    // Phonon eigenvectors
    thrust::complex<float> eigvec[3 * natom];
    for (unsigned ia = 0; ia < natom; ia++) {
        eigvec[ia * 3 + 0] = thrust::complex<float>(matSys.modes[imode-1].disp[ia][0].real(), matSys.modes[imode-1].disp[ia][0].imag());
        eigvec[ia * 3 + 1] = thrust::complex<float>(matSys.modes[imode-1].disp[ia][1].real(), matSys.modes[imode-1].disp[ia][1].imag());
        eigvec[ia * 3 + 2] = thrust::complex<float>(matSys.modes[imode-1].disp[ia][2].real(), matSys.modes[imode-1].disp[ia][2].imag());
    }
    thrust::complex<float>* eigvec_dev;
    CUDA_CALL( cudaMalloc((void**) &eigvec_dev, 3 * natom * sizeof(thrust::complex<float>)) );
    CUDA_CALL( cudaMemcpy(eigvec_dev, eigvec, 3 * natom * sizeof(thrust::complex<float>), cudaMemcpyHostToDevice) );

    // Time step
    float delta_omega_t = 2 * M_PI / nstep_per_cycle;

    /**************************************************************
     * Looping through frames (and the relevant variables)
     *************************************************************/

    // Phase
    thrust::complex<float> phase;

    // Displaced sphere positions
    float3* sphere_displaced_pos_dev;
    CUDA_CALL( cudaMalloc((void**) &sphere_displaced_pos_dev, (natom+1) * sizeof(float3)) );
    CUDA_CALL( cudaMemcpy(sphere_displaced_pos_dev, sphere_pos, (natom+1) * sizeof(float3), cudaMemcpyHostToDevice) );

    // Output char array
    char video[3 * width * height];
    char* video_dev;
    CUDA_CALL( cudaMalloc((void**) &video_dev, 3 * width * height * sizeof(char)) );

    // All level of rays (from each pixel)
    ray* allRays[MAX_RAY_DEPTH + 1];
    for (unsigned int ird = 0; ird < MAX_RAY_DEPTH + 1; ird ++) {
        CUDA_CALL( cudaMalloc((void**) &allRays[ird], TwoToThePowOf(ird) * width * height * sizeof(ray)) );
        CUDA_CALL( cudaMemset(allRays[ird], 0, TwoToThePowOf(ird) * width * height * sizeof(ray)) );
    }
    // The initial raydirs were shot all from each pixel, so remain unchanged throughout simulation
    cuda_initialize_firstLevel_rays_dir_kernel<<<512, 512>>>(width, height, invWidth, invHeight, angle, aspectratio, allRays[0]);

    // Looping through frames
    for (int it = 0; it < nstep_per_cycle; it++) {

        std::cout << "Calculating frame: " << it+1 << " / " << nstep_per_cycle << std::endl;
        phase = thrust::exp(thrust::complex<float>(0, -it*delta_omega_t));
        cuda_get_sphere_pos_kernel<<<1, 256>>>(natom, sphere_pos_dev, amplitude, phase, eigvec_dev, sphere_displaced_pos_dev);    
        
        // Forward pass of rays (find intersecting spheres and trace reflection and refraction rays)
        for (unsigned int ird = 0; ird < MAX_RAY_DEPTH; ird ++) {
            cuda_find_intersecting_sphere_kernel<<<4096, 512>>> \
                (width, height, natom+1, species_sphere_prop_dev, sphere_type_dev, sphere_displaced_pos_dev, ird, allRays[ird]);
            cuda_trace_reflection_and_refraction_kernel<<<4096, 512>>> \
                (width, height, ird, allRays[ird], allRays[ird+1]);
        }

        // Special treatment for the last level of rays
        cuda_find_intersecting_sphere_kernel<<<4096, 512>>> \
            (width, height, natom+1, species_sphere_prop_dev, sphere_type_dev, sphere_displaced_pos_dev, MAX_RAY_DEPTH, allRays[MAX_RAY_DEPTH]);
        cuda_compute_last_level_Color_kernel<<<4096, 512>>> \
            (width, height, natom+1, species_sphere_prop_dev, sphere_type_dev, sphere_displaced_pos_dev, allRays[MAX_RAY_DEPTH]);

        // Backward pass of rays (compute ray colors)
        for (int ird = MAX_RAY_DEPTH - 1; ird >= 0; ird--) {
            cuda_compute_Color_kernel<<<4096, 512>>> \
                (width, height, natom+1, species_sphere_prop_dev, sphere_type_dev, ird, allRays[ird], allRays[ird+1]);
        }

        // Collect all the ray colors in the first level
        cuda_collect_pixels<<<4096, 512>>>(width, height, allRays[0], video_dev);
        CUDA_CALL( cudaMemcpy(video, video_dev, 3 * width * height * sizeof(char), cudaMemcpyDeviceToHost) );

        // write to tmp file
        tmpFile.write(video, 3 * width * height);
    }

    // Free resources
    tmpFile.close();
    CUDA_CALL( cudaFree(species_sphere_prop_dev) );
    CUDA_CALL( cudaFree(sphere_type_dev) );
    CUDA_CALL( cudaFree(sphere_pos_dev) );
    CUDA_CALL( cudaFree(eigvec_dev) );
    CUDA_CALL( cudaFree(sphere_displaced_pos_dev) );
    for (unsigned int ird = 0; ird < MAX_RAY_DEPTH + 1; ird ++) {
        CUDA_CALL( cudaFree(allRays[ird]) );
    }

    // Rendering using ffmpeg
    std::cout << "Outputting video: " + matSys.prefix + "_" + std::to_string(imode) + "_gpu.mp4" << std::endl;
    std::string command = "ffmpeg -hide_banner -loglevel error -stream_loop "+std::to_string(nloop) \
                          + " -y -f rawvideo -pixel_format gbrp -video_size " + std::to_string(width) +"x" +std::to_string(height) \
                          + " -i ./results/tmp.bin -c:v h264 -pix_fmt yuv420p " \
                          + "results/" + matSys.prefix + "_" + std::to_string(imode) + "_gpu.mp4";
    std::system(command.c_str());
    std::remove("./results/tmp.bin");
}