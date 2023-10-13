/**************************************
 * Main driver program for the CPU part
 **************************************/
#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <fstream>
#include <vector>
#include <iostream>
#include <cassert>
#include <unordered_map>
#include "raytracer.h"
#include "materialsystem.h"

std::unordered_map<std::string, float> atomSize = {
    {"H", 0.05},
    {"C", 0.09}
};

std::unordered_map<std::string, Vec3f> atomColor = {
    {"H", Vec3f(0.90, 0.76, 0.46)},
    {"C", Vec3f(0.65, 0.77, 0.97)}
};

// t0, t1: the distances of the two intersections of a shadow ray penetrating a sphere from the ray origin
/********************************************************
 * This function is planned to become a device function
 ********************************************************/
bool Sphere::intersect(const Vec3f &rayorig, const Vec3f &raydir, float &t0, float &t1) const {
    Vec3f l = center - rayorig;
    float tca = l.dot(raydir);
    if (tca < 0) return false;
    float d2 = l.dot(l) - tca * tca;
    if (d2 > radius2) return false;
    float thc = sqrt(radius2 - d2);
    t0 = tca - thc;
    t1 = tca + thc;
        
    return true;
}

/*********************************************************
 * This function is planned to become a device function
 ********************************************************/
float mix(const float &a, const float &b, const float &mix) {
    return b * mix + a * (1 - mix);
}

/*************************************************************************
 * Since trace() is called recursively with non-predetermined depth
 * (although with a predefined maximum depth of 5), it is not reasonable to 
 * parallelize trace() at the pixel level. Instead, I should parallelize
 * trace() dynamically at the ray level. Each ray has at most two rays
 * (reflection and refraction rays) to trace, so we can imagine this 
 * process as growing a binary tree. I will write a loop that runs the 
 * trace kernel 5 times, where at each loop I will save the addition rays
 * that we need to trace to a thrust vector for the next loop.
 *************************************************************************/
Vec3f trace(
    const Vec3f &rayorig,
    const Vec3f &raydir,
    const std::vector<Sphere> &spheres,
    const int &depth)
{
    float tnear = INFINITY;         // record the nearest intersection of the shadow ray with a sphere
    const Sphere* sphere = NULL;

    // find the intersection(s) of this ray with the sphere in the scene
    for (unsigned i = 0; i < spheres.size(); ++i) {
        float t0 = INFINITY, t1 = INFINITY;
        if (spheres[i].intersect(rayorig, raydir, t0, t1)) {
            if (t0 < 0) t0 = t1;
            if (t0 < tnear) {
                tnear = t0;
                sphere = &spheres[i];
            }
        }
    }

    // if there's no intersection return black or background color
    if (!sphere) return Vec3f(2);
    Vec3f surfaceColor = 0; // color of the ray/surfaceof the object intersected by the ray
    Vec3f phit = rayorig + raydir * tnear; // point of intersection
    Vec3f nhit = phit - sphere->center; // normal at the intersection point
    nhit.normalize(); // normalize normal direction
    // If the normal and the view direction are not opposite to each other
    // reverse the normal direction. That also means we are inside the sphere so set
    // the inside bool to true. Finally reverse the sign of IdotN which we want
    // positive.
    float bias = 1e-4; // add some bias to the point from which we will be tracing
    bool inside = false;
    if (raydir.dot(nhit) > 0) nhit = -nhit, inside = true;
    if ((sphere->transparency > 0 || sphere->reflection > 0) && depth < MAX_RAY_DEPTH) {
        float facingratio = -raydir.dot(nhit);
        // change the mix value to tweak the effect
        float fresneleffect = mix(pow(1 - facingratio, 3), 1, 0.1);
        // compute reflection direction (not need to normalize because all vectors
        // are already normalized)
        Vec3f refldir = raydir - nhit * 2 * raydir.dot(nhit);
        refldir.normalize();
        Vec3f reflection = trace(phit + nhit * bias, refldir, spheres, depth + 1);
        Vec3f refraction = 0;
        // if the sphere is also transparent compute refraction ray (transmission)
        if (sphere->transparency) {
            float ior = 1.1, eta = (inside) ? ior : 1 / ior; // are we inside or outside the surface?
            float cosi = -nhit.dot(raydir);
            float k = 1 - eta * eta * (1 - cosi * cosi);
            Vec3f refrdir = raydir * eta + nhit * (eta *  cosi - sqrt(k));
            refrdir.normalize();
            refraction = trace(phit - nhit * bias, refrdir, spheres, depth + 1);
        }
        // the result is a mix of reflection and refraction (if the sphere is transparent)
        surfaceColor = (
            reflection * fresneleffect +
            refraction * (1 - fresneleffect) * sphere->transparency) * sphere->surfaceColor;
    }
    else {
        // it's a diffuse object, no need to raytrace any further
        // this part concerns the light emissions from other spheres onto the current sphere's hit point
        /***********************************************************************************************
        * Since in my case only one sphere (the light source) is emitting light, I can remove the 
        * following for loop and just check for the light source.
        ***********************************************************************************************/
        for (unsigned i = 0; i < spheres.size(); ++i) {
            if (spheres[i].emissionColor.x > 0) {
                // this is a light
                Vec3f transmission = 1;
                Vec3f lightDirection = spheres[i].center - phit;
                lightDirection.normalize();
                for (unsigned j = 0; j < spheres.size(); ++j) {
                    if (i != j) {
                        float t0, t1;
                        if (spheres[j].intersect(phit + nhit * bias, lightDirection, t0, t1)) {
                            transmission = 0;
                            break;
                        }
                    }
                }
                surfaceColor += sphere->surfaceColor * transmission *
                std::max(float(0), nhit.dot(lightDirection)) * spheres[i].emissionColor;
            }
        }
    }  
    return surfaceColor + sphere->emissionColor;
}

// Reference: https://stackoverflow.com/questions/24228728/create-video-from-array-of-pixel-values-in-c
// Rendering makes use of ffmpeg
void renderVideo(const MaterialSystem matSys,
                 const unsigned imode,
                 const unsigned width,
                 const unsigned height,
                 const unsigned nstep_per_cycle,
                 const unsigned nloop,
                 const float    amplitude)
{
    // Settings
    Vec3f *image = new Vec3f[width * height];
    float invWidth = 1 / float(width), invHeight = 1 / float(height);
    float fov = 30, aspectratio = width / float(height);
    float angle = tan(M_PI * 0.5 * fov / 180.);
    float delta_omega_t = 2 * M_PI / nstep_per_cycle;

    // variable initialization
    Vec3f *pixel;
    std::complex<float> phase;
    std::string spec;
    std::ofstream tmpFile("./results/tmp.bin", std::ios::binary);
    char video[3 * width * height];
    std::vector<Sphere> spheres;

    // Loop over frames
    for (int it = 0; it < nstep_per_cycle; it++) {
        std::cout << "Calculating frame: " << it+1 << " / " << nstep_per_cycle << std::endl;
        spheres.clear();
        pixel = image;
        phase = std::exp(std::complex<float>(0, -it*delta_omega_t));

        // compute atom positions and add spheres to the scene
        for (unsigned ia = 0; ia < matSys.natom; ia++) {
            spec = matSys.atoms[ia].species;
            auto [x, y, z] = matSys.atoms[ia].pos_cart_alat;
            auto [dx, dy, dz] = matSys.modes[imode-1].disp[ia];
            dx *= std::complex<float>(amplitude, 0) * phase;
            dy *= std::complex<float>(amplitude, 0) * phase;
            dz *= std::complex<float>(amplitude, 0) * phase;
            spheres.push_back(Sphere(Vec3f(x + dx.real(), y + dy.real(), z + dz.real()), atomSize[spec], atomColor[spec], 0.1, 0.4));
        }
        spheres.push_back(Sphere(Vec3f( 0.0, -20, -30), 3, Vec3f(0.0, 0.0, 0.0), 0, 0.0, Vec3f(3))); // light

        // trace rays and render
        /************************************************************************************
         * Useful reference:
         * https://dc.ewu.edu/cgi/viewcontent.cgi?referer=&httpsredir=1&article=1093&context=theses
         * This for loop can be parallelized at the pixel level using a kernel
         * For the parallelization of trace(),  see the block comments in raytracer.cpp/.h
        ************************************************************************************/
        for (unsigned y = 0; y < height; ++y) {
            for (unsigned x = 0; x < width; ++x, ++pixel) {
                float xx = (2 * ((x + 0.5) * invWidth) - 1) * angle * aspectratio;
                float yy = (1 - 2 * ((y + 0.5) * invHeight)) * angle;
                Vec3f raydir(xx, yy, -1);
                raydir.normalize();
                *pixel = trace(Vec3f(0), raydir, spheres, 0);
            }
        }

        // save pixelated picture
        /************************************************************************************
        * This part can be parallelized with a kernel where each thread takes care of a pixel
        * (or an RGB channel of a pixel)
        ************************************************************************************/
        for (int ip = 0; ip < width * height; ip++) {
            video[ip] = (unsigned char)(std::min(float(1), image[ip].y) * 255);
            video[ip+width*height] = (unsigned char)(std::min(float(1), image[ip].z) * 255);
            video[ip+2*width*height] = (unsigned char)(std::min(float(1), image[ip].x) * 255);
        }

        // write to tmp file
        tmpFile.write(video, 3 * width * height);
    }

    tmpFile.close();

    // Rendering using ffmpeg
    std::cout << "Outputting video: " + matSys.prefix + "_" + std::to_string(imode) + "_cpu.mp4" << std::endl;
    std::string command = "ffmpeg -hide_banner -loglevel error -stream_loop "+std::to_string(nloop) \
                          + " -y -f rawvideo -pixel_format gbrp -video_size " + std::to_string(width) +"x" +std::to_string(height) \
                          + " -i ./results/tmp.bin -c:v h264 -pix_fmt yuv420p " \
                          + "results/" + matSys.prefix + "_" + std::to_string(imode) + "_cpu.mp4";
    std::system(command.c_str());
    std::remove("./results/tmp.bin");
}