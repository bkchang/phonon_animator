/**************************************
 * Main driver program
**************************************/
#include <string>
#include <chrono>
#include "materialsystem.h"
#include "raytracer.h"
#include "raytracer.cuh"

int main() {

    // User input
    std::string prefix, imode, width, height, nframe, nloop;
    std::cout << "System to animate? (Options: Benz | Naph | Anth | Tetr | Biph) " << std::endl;
    std::getline(std::cin, prefix);
    MaterialSystem matSys(prefix);
    std::cout << "Which mode? (Enter between 1-" << std::to_string(matSys.natom*3) << ")" << std::endl;
    std::getline(std::cin, imode);
    std::cout << "Width of video in pixel? (Suggested: 640 ~ 1920)" << std::endl;
    std::getline(std::cin, width);
    std::cout << "Height of video in pixel? (Suggested: 480 ~ 1080)" << std::endl;
    std::getline(std::cin, height);
    std::cout << "Number of frames per oscillation cycle? (Suggested: 20)" << std::endl;
    std::getline(std::cin, nframe);
    std::cout << "Number of repeated loops? (Suggested: 4) This does not affect the CPU/GPU calculations" << std::endl;
    std::getline(std::cin, nloop);
    //matSys.printSystemInfo(); // for debugging

    std::cout << "[CPU rendering]" << std::endl;
    auto startTime = std::chrono::high_resolution_clock::now();
    renderVideo(matSys, std::stoul(imode), 
                        std::stoul(width),
                        std::stoul(height),
                        std::stoul(nframe),
                        std::stoul(nloop));
    auto endTime = std::chrono::high_resolution_clock::now();
    auto duration_cpu = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();
    std::cout << "Done. Time used: " << duration_cpu << " (ms)" << std::endl;

    std::cout << std::endl;

    std::cout << "[GPU rendering]" << std::endl;
    startTime = std::chrono::high_resolution_clock::now();
    renderVideo_gpu(matSys, std::stoul(imode), 
                            std::stoul(width),
                            std::stoul(height),
                            std::stoul(nframe),
                            std::stoul(nloop));
    endTime = std::chrono::high_resolution_clock::now();
    auto duration_gpu = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();
    std::cout << "Done. Time used: " << duration_gpu << " (ms)" << std::endl;
    
    std::cout << "-------------------" << std::endl;
    std::printf("GPU acceleration factor: %.2fx\n", static_cast<float>(duration_cpu) / static_cast<float>(duration_gpu));

    return 0;
}