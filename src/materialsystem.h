#include <vector>
#include <string>
#include <complex>
#include <array>

#ifndef MATERIALSYSTEM_H
#define MATERIALSYSTEM_H

class MaterialSystem
{
    private:
        void readFiles();

    public:
        struct Atom {
            std::string species;
            float pos_cart_alat[3];
        };
        struct Phonon {
            float energy_cm_1;
            std::vector<std::array<std::complex<float>, 3>> disp;
        };

        unsigned natom;
        float crystal_axes_cart_alat[3][3];
        std::vector<Atom> atoms;
        std::vector<Phonon> modes;
        std::string prefix, file_out, file_matdyn;

        MaterialSystem(std::string input_prefix);

        void printSystemInfo();
};

#endif