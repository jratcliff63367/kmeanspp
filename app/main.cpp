#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <random>

#include "ScopedTime.h"
#define ENABLE_KMEANS_IMPLEMENTATION 1
#include "kmeanspp.h"

void testKmeans(uint32_t pointCount,uint32_t k,bool useKdTree,bool useThreading)
{
    float *points = new float[pointCount * 3];
    std::random_device rand_device;
    uint64_t seed = 0; //rand_device();
    // Using a very simple PRBS generator, parameters selected according to
    // https://en.wikipedia.org/wiki/Linear_congruential_generator#Parameters_in_common_use
    std::linear_congruential_engine<uint64_t, 6364136223846793005, 1442695040888963407, UINT64_MAX> rand_engine(seed);
    std::uniform_int_distribution<size_t> uniform_generator(0, 10000);
    for (uint32_t i = 0; i < pointCount * 3; i++)
    {
        points[i] = (float)uniform_generator(rand_engine);
    }
    kmeans::Kmeans *kpp = kmeans::Kmeans::create();
    uint32_t resultPointCount;
    printf("Running Kmeans against:%d input points.\n", pointCount);
    {
        ScopedTime st("Kmeans Time");
        kmeans::Kmeans::Parameters p;
        p.mPoints = points;
        p.mPointCount = pointCount;
        p.mMaxPoints = k;
        //        p.mMaximumPlusPlusCount = pointCount; // p.mMaxPoints*4;
        p.mMaximumPlusPlusCount = pointCount;

        p.mUseKdTree = useKdTree;
        p.mUseThreading = useThreading;
        const float *results = kpp->compute(p, resultPointCount);
        for (uint32_t i=0; i<resultPointCount; i++)
        {
            const float *mp = &results[i*3];
            printf("[%d]=(%0.2f,%0.2f,%0.2f)\n", i+1, mp[0], mp[1], mp[2]);
        }
    }
    kpp->release();
}

int main(int argc,const char **argv)
{
    testKmeans(1000,32,false,false);
//    testKmeans(1000,32,true,false);
//    testKmeans(1000, 32, false, true);
    testKmeans(1000, 32, true, true);
}
