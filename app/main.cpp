#include <stdio.h>
#include <stdlib.h>
#include <random>

#include "ScopedTime.h"
#define ENABLE_KMEANS_IMPLEMENTATION 1
#include "kmeans.h"


int main(int argc,const char **argv)
{
    uint32_t pointCount = 10000;
    float *points = new float[pointCount*3];
    std::random_device rand_device;
    uint64_t seed = 0; //rand_device();
    // Using a very simple PRBS generator, parameters selected according to
    // https://en.wikipedia.org/wiki/Linear_congruential_generator#Parameters_in_common_use
    std::linear_congruential_engine<uint64_t, 6364136223846793005, 1442695040888963407, UINT64_MAX> rand_engine(seed);
    std::uniform_int_distribution<size_t> uniform_generator(-1000,1000);
    for (uint32_t i=0; i<pointCount*3; i++)
    {
        points[i] = (float) uniform_generator(rand_engine);
    }
    kmeans::Kmeans *kpp = kmeans::Kmeans::create();
    uint32_t resultPointCount;
    printf("Running Kmeans against:%d input points.\n", pointCount);
    {
        ScopedTime st("Kmeans Time");
        const float *results = kpp->compute(points, pointCount, 1000, resultPointCount);
    }
    kpp->release();
}
