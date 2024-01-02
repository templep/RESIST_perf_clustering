# Measurement protocol

## Software Systems

We consider 8 software systems. We choose them because they are open-source, well-known in various fields and already studied in the literature: gcc, ImageMagick, lingeling, nodeJS, poppler, SQLite, x264 and xz.
We choose these systems because they handle different types of input data, allowing us to draw as general conclusions as possible.
For each software system, we use a unique private server with the same configuration running over the same operating system. The configurations of the running environments are available at: https://anonymous.4open.science/r/df319578-8767-47b0-919d-a8e57eb67d25/replication/Environments.md
We download and compile a unique version of the system. 
All performance are measured with this version of the software.

## Configuration Options

To select the configuration options, we read the documentation of each system.
We manually extracted the options affecting the performance of the system according to the documentation.
We then sampled #C configurations by using random sampling. We checked the uniformity of the different option values with a Kolmogorov-Smirnov test applied to each configuration option. Options and tests results are available at : https://anonymous.4open.science/r/df319578-8767-47b0-919d-a8e57eb67d25/results/others/configs/sampling.md.

## Inputs

For each system, we selected a different set of input data: for gcc, PolyBench v3.1 ; for ImageMagick, a sample of ImageNet images (from 1.1 kB to 7.3 MB); for ingeling, the 2018 SAT competition's benchmark; for nodeJS, its test suite;  for poppler, the Trent Nelson's PDF Collection; for SQLite, a set of generated TPC-H databases (from 10 MB to 6 GB); for x264, the YouTube User General Content dataset of videos (from 2.7 MB to 39.7 GB); for xz, the Silesia corpus.
We choose them because these are large and freely available datasets of inputs, well-known in their field and already used by researchers or practitioners.

## Performance properties

For each system, we systematically executed all the configurations of C on all the inputs of I. For the #M resulting executions, we measured as many performance properties as possible: for gcc, ctime and exec the times needed to compile and execute a program and the size of the binary; for ImageMagick, the time to apply a Gaussian blur to an image and the size of the resulting image; for lingeling, the number of conflicts  and reductions found in 10 seconds of execution; for nodeJS,  the number of operations per second (ops) executed by the script; for poppler, the time needed to extract the images of the pdf, and the size of the images; for SQLite, the time needed to answer 15 different queries q1-q15; for x264, the bitrate (the average amount of data encoded per second), the cpu usage (percentage), the average number of frames encoded per second (fps), the size of the compressed video and the elapsed time; for xz, the size of the compressed file, and the time needed to compress it.

## Reproducibility

To allow researchers to easily replicate the measurement process, we provide a docker container for each system (see the links in the Docker column).
We also publish the resulting datasets online (see the links in the Data column) and in https://anonymous.4open.science/r/df319578-8767-47b0-919d-a8e57eb67d25/replication/README.md

