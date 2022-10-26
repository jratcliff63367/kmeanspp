#pragma once

#include <stdint.h>

namespace kmeans
{

class Kmeans
{
public:
    static Kmeans *create(void);

    virtual const float *compute(const float *sourcePoints,
        uint32_t pointCount,
        uint32_t maxPoints,
        uint32_t &resultPointCount) = 0;

    virtual void release(void) = 0;
protected:
    virtual ~Kmeans(void)
    {
    }
};

}

#if ENABLE_KMEANS_IMPLEMENTATION

#include <random>
#include <vector>

#pragma warning(disable:4267)

#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <random>
#include <tuple>
#include <type_traits>
#include <vector>

namespace kmeans
{

class Point3
{
public:

    float distanceSquared(const Point3 &p) const
    {
        float dx = x-p.x;
        float dy = y-p.y;
        float dz = z-p.z;
        return dx*dx + dy *dy + dz *dz;
    }

    float   x;
    float   y;
    float   z;
};

using Point3Vector = std::vector< Point3 >;
using ClusterVector = std::vector< uint32_t >;
using DistanceVector = std::vector< float >;

class KmeansImpl : public Kmeans
{
public:
    KmeansImpl(void)
    {
    }

    virtual ~KmeansImpl(void)
    {
    }

    virtual const float *compute(const float *sourcePoints,
        uint32_t pointCount,
        uint32_t maxPoints,
        uint32_t &resultPointCount) final
    {
        const float *ret = nullptr;

        mMeans.clear();
        mMeans.reserve(maxPoints);
        mClusters.clear();
        mClusters.reserve(maxPoints);
        mData.resize(pointCount);
        memcpy(&mData[0],sourcePoints,sizeof(float)*3*pointCount);
        initializeClusters();


        return ret;
    }

    virtual void release(void) final
    {
        delete this;
    }

    void initializeClusters(void)
    {
        std::random_device rand_device;
        uint64_t seed = rand_device();
        // Using a very simple PRBS generator, parameters selected according to
        // https://en.wikipedia.org/wiki/Linear_congruential_generator#Parameters_in_common_use
        std::linear_congruential_engine<uint64_t, 6364136223846793005, 1442695040888963407, UINT64_MAX> rand_engine(seed);
        // Select first mean at random from the set
        {
            std::uniform_int_distribution<size_t> uniform_generator(0, mData.size() - 1);
            mMeans.push_back(mData[uniform_generator(rand_engine)]);
        }

        for (uint32_t count = 1; count < mK; ++count) 
        {
            // Calculate the distance to the closest mean for each data point
            auto distances = closestDistance(mMeans, mData);
            // Pick a random point weighted by the distance from existing means
            // TODO: This might convert floating point weights to ints, distorting the distribution for small weights
            std::discrete_distribution<size_t> generator(distances.begin(), distances.end());
            mMeans.push_back(mData[generator(rand_engine)]);
        }

    }

    DistanceVector closestDistance(const Point3Vector &means,const Point3Vector &data) 
    {
        DistanceVector distances;
        distances.reserve(data.size());
        for (auto& d : data) 
        {
            float closest = d.distanceSquared(means[0]);
            for (auto& m : means) 
            {
                float distance = d.distanceSquared(m);
                if (distance < closest)
                {
                    closest = distance;
                }
            }
            distances.push_back(closest);
        }
        return distances;
    }

    uint32_t        mK{32};     // Maximum number of mean values to produce
    Point3Vector    mData;      // Input data
    Point3Vector    mMeans;     // Means
    ClusterVector   mClusters;  // Which cluster each source data point is in
};

Kmeans *Kmeans::create(void)
{
    auto ret = new KmeansImpl;
    return static_cast< Kmeans *>(ret);
}


}
#endif