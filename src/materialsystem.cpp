#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <fstream>
#include <iterator>
#include <complex>
#include <array>
#include <iomanip>

#include "materialsystem.h"

#define SYSTEM_Z_SHIFT 3

MaterialSystem::MaterialSystem(std::string input_prefix) {
    prefix = input_prefix;
    file_out = "./data/" + prefix + "/" + prefix + ".scf.out";
    file_matdyn = "./data/" + prefix + "/" + prefix + ".matdyn.modes";
    readFiles();
}

void MaterialSystem::readFiles() {
    std::string line;
    std::ifstream File;
    std::istringstream iss;
    std::vector<std::string> tokens;

    /*
        scf.out
        This file contains the cell and atomic information
    */
    File.open(file_out);
    // natom
    while(std::getline(File, line)) {
        if (line.find("number of atoms/cell", 0) != std::string::npos)
            break;
    }
    iss.clear();
    iss.str(line);
    tokens.assign(std::istream_iterator<std::string>{iss}, {});
    natom = std::stoi(tokens.back());
    // crystal axes
    while(std::getline(File, line)) {
        if (line.find("crystal axes:", 0) != std::string::npos)
            break;
    }
    for (int r = 0; r < 3; r++) {
        std::getline(File, line);
        iss.clear();
        iss.str(line);
        tokens.assign(std::istream_iterator<std::string>{iss}, {});
        crystal_axes_cart_alat[r][0] = std::stod(tokens[3]);
        crystal_axes_cart_alat[r][1] = std::stod(tokens[4]);
        crystal_axes_cart_alat[r][2] = std::stod(tokens[5]); 
    }
    // atomic positions
    float center[3] = {0.0, 0.0, 0.0};
    while(std::getline(File, line)) {
        if (line.find("site n.", 0) != std::string::npos)
            break;
    }
    for (int ia = 0; ia < natom; ia++) {
        std::getline(File, line);
        iss.clear();
        iss.str(line);
        tokens.assign(std::istream_iterator<std::string>{iss}, {});
        Atom tmp;
        tmp.species = tokens[1];
        tmp.pos_cart_alat[0] = std::stod(tokens[6]);
        tmp.pos_cart_alat[1] = std::stod(tokens[7]);
        tmp.pos_cart_alat[2] = std::stod(tokens[8]);
        center[0] += tmp.pos_cart_alat[0];
        center[1] += tmp.pos_cart_alat[1];
        center[2] += tmp.pos_cart_alat[2];
        atoms.push_back(tmp);
    }
    // Shifting the atoms to accommodate camera angle and scene
    File.close();
    center[0] /= natom;
    center[1] /= natom;
    center[2] /= natom;
    for (int ia = 0; ia < natom; ia++) {
        atoms[ia].pos_cart_alat[0] -= center[0];
        atoms[ia].pos_cart_alat[1] -= center[1];
        atoms[ia].pos_cart_alat[2] -= center[2] + SYSTEM_Z_SHIFT;
    }
    /*
        matdyn.modes
        This file contains the vibrational information
    */
    File.open(file_matdyn);
    for (int i = 0; i < 4; i++)
        std::getline(File, line);
    for (int imode = 0; imode < natom * 3; imode++){
        Phonon ph;
        std::getline(File, line);
        iss.clear();
        iss.str(line);
        tokens.assign(std::istream_iterator<std::string>{iss}, {});
        ph.energy_cm_1 = std::stod(tokens[7]);
        for (int ia = 0; ia < natom; ia++){
            std::getline(File, line);
            iss.clear();
            iss.str(line);
            tokens.assign(std::istream_iterator<std::string>{iss}, {});
            std::array<std::complex<float>, 3> tmp;
            tmp[0] = std::complex(std::stod(tokens[1]), std::stod(tokens[2]));
            tmp[1] = std::complex(std::stod(tokens[3]), std::stod(tokens[4]));
            tmp[2] = std::complex(std::stod(tokens[5]), std::stod(tokens[6]));
            ph.disp.push_back(tmp);
        }
        modes.push_back(ph);
    }
}

void MaterialSystem::printSystemInfo() {
    std::cout << std::setprecision(7) << std::fixed;
    std::cout << "Number of atoms " << natom << std::endl;
    std::cout << "Cell vectors (cryst. coord., alat): " << std::endl;
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++)
            std::cout << crystal_axes_cart_alat[i][j] << "\t";
        std::cout << std::endl;
    }
    std::cout << "First 3 atoms (cryst. coord., alat): " << std::endl;
    for (int ia = 0; ia < 3; ia++) {
        std::cout << atoms[ia].species << "\t" << \
                     atoms[ia].pos_cart_alat[0] << "\t" << \
                     atoms[ia].pos_cart_alat[1] << "\t" << \
                     atoms[ia].pos_cart_alat[2] << "\t" << std::endl;
    }
    std::cout << "First 3 atoms (cryst. coord., alat): " << std::endl;
    for (int ia = 0; ia < 3; ia++) {
        std::cout << atoms[ia].species << "\t" << \
                     atoms[ia].pos_cart_alat[0] << "\t" << \
                     atoms[ia].pos_cart_alat[1] << "\t" << \
                     atoms[ia].pos_cart_alat[2] << "\t" << std::endl;
    }
    std::cout << "First 5 modes and their first and last displacement vectors: " << std::endl;
    for (int imode = 0; imode < 5; imode++) {
        std::cout << "Mode " << imode+1 << ": " << modes[imode].energy_cm_1 << " (cm-1)" << std::endl;
        std::cout << modes[imode].disp[0][0] << " " \
                  << modes[imode].disp[0][1] << " " \
                  << modes[imode].disp[0][2] << std::endl;
        std::cout << modes[imode].disp[natom-1][0] << " " \
                  << modes[imode].disp[natom-1][1] << " " \
                  << modes[imode].disp[natom-1][2] << std::endl;  
    }
}