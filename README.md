# phonon-animator

## Project description

This project implements a program that animates the phonon vibrations of materials using a raytracing renderer. The phonon data were precalculated using the well-used quantum chemistry software [Quantum ESPRESSO (QE)](https://www.quantum-espresso.org). Here I provide the phonon data of 5 organic molecular crystal systems (benzene, naphthalene, anthracene, tetracene, and biphenyl) that I actually computed for my research in `./data/`. A material has `3 x (number of atoms)` of phonon modes, so when running the program, the user should specify not only the system, but also the index of the phonon mode that is to be animated. The program renders each frame of the animation and export it into a video using `ffmpeg`.

## Usage instructions

- Installation: the zipped file is already pre-compiled. But if one wants to build the project on Titan, I have provided a makefile, and by running `make` the project should build completely and automatically (ignore the warnings, which does not affect the goal and conclusion of this project). Remember that `ffmpeg` must be installed in advance. Also make sure that `./obj/`, `./bin/`, `./result/` directories exist.
- Running the program: No script is needed for running the program. Just run `./bin/animation` (or the program will be automatically executed after running `make`), then there will be several prompts to ask you to enter the desired settings for the video. In particular:
  1. Enter the system you would like to animate, options include: "Benz", "Naph", "Anth", "Tetr", "Biph". Do not include the quotes.
  2. Enter the phonon mode index of interest, just enter an integer number within the range given by the prompt.
  3. Enter the width of the video in pixel. 640 is a good choice to test the difference between CPU and GPU performance. >1920 is not ideal as it is very slow for the CPU part and may cause segmentation fault.
  4. Enter the height of the video in pixel. 480 is good. >1080 might be too large.
  5. Enter the number of frames for a phonon oscillation cycle. 30 is a good number for animation. More frames will take longer to calculale and make the animation look slower but smoother.
  6. Enter the number of loops of the animation, i.e. how many times you would like the frames to repeat in the final video. Any number >=4 should be good to check out the animation. This looping is done by `ffmpeg` and is very fast, and should not affect our conclusion on the CPU/GPU code performance.
  7. Wait for the code to finish. The code will show rendering progress as well as how much the GPU code accelerates the calculations.

## Results

Videos are rendered and saved with names in the format of `./results/${system}_${mode}_${cpu/gpu}.mp4`, where `${system}` is the prefix of the material system, `${mode}` is the index of the phonon, and `${cpu/gpu}` denotes whether it is rendered by the CPU code or the GPU code. Some example outputs from different systems, phonons modes, and resolutions are already provided in the `./results` directory.

## Performance

I hard-coded the number of blocks (4096) and threads (512) for each kernel in my implementation, and got 8x~14x acceleration in the cases that I tested. Increasing the number of blocks can potentially lead to a little better acceleration, but the improvement is expected to plateau due to communication overhead.

## Implementation details

The presented ray-tracing algorithm has the logic as follows:
1. The color of a pixel on the canvas is determined by a ray shooting from that pixel to the camera.
2. Each ray's color is determined by two cases:
    - If the ray's reverse doesn't hit any object, it's the color of the background
    - If the ray's reverse does hit an object, then its color is determined by the back-traced reflection ray and refraction ray.

So we can imagine that for each pixel, we are growing a binary tree of rays based on the condition of whether the ray hit an object.
```
           ray
          /    \
reflection      refraction
    ray            ray
```
We can grow this binary tree indefinitely. Here our set maximum depth of the tree is 5. To know the color of a parent ray, the color of the children rays must be known.

In the original CPU implementation, we grow the tree recursively and conditionally (on whether it hits an object) to make sure that we are computing the rays that we actually need to compute, and to reduce computation time.

In the GPU implementation, to facilitate the parallelization and clarity of the code, I sacrifice some memory space and grow a "perfect binary tree", i.e. regardless of whether the ray hit an object, I will go to its (fake) reflection and refraction ray (although their information may not be used later).

```
Ray binary tree for each pixel

                  ray               -------level 1
                 /    \
                /      \
               /        \
              /          \
           ray            ray       -------level 2
          /   \          /   \
       ray     ray    ray     ray   -------level 3

                   ...
```
With this structure, I know exactly how many rays there are at each level, so I can write a loop to go through each level of rays and find their children (the forward pass). After I construct the entire binary tree (find all the rays), I compute the colors of the rays in last level, and write a loop to move back up to compute the colors of the rays in upper levels (backward pass). The GPU code is thus parallelized with repect to rays.