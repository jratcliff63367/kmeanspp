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

        mK = maxPoints;
        mMeans.clear();
        mMeans.reserve(mK);

        mClusters.resize(pointCount);

        mData.resize(pointCount);
        memcpy(&mData[0],sourcePoints,sizeof(float)*3*pointCount);

        initializeClusters();

        uint32_t count = 0;
        uint32_t maxIterations = 100;
        mOldMeans = mMeans;
        mOldOldMeans = mMeans;
        do 
        {
            calculateClusters();
            mOldOldMeans = mOldMeans;
            mOldMeans = mMeans;
            calculateMeans();
            count++;
            if ( sameMeans(mMeans,mOldMeans))
            {
                break;
            }
            if (sameMeans(mMeans, mOldOldMeans))
            {
                break;
            }
        } while ( count < maxIterations );

        resultPointCount = mK;
        ret = &mMeans[0].x;

        return ret;
    }

    bool sameMeans(const Point3Vector &a,const Point3Vector &b)
    {
        bool ret = false;

        if ( memcmp(&a[0],&b[0],sizeof(Point3)*mK) == 0 )
        {
            ret = true;
        }

        return ret;
    }

    void calculateMeans(void)
    {
        std::vector< uint32_t > counts;
        counts.resize(mK);
        memset(&counts[0],0,sizeof(uint32_t)*mK);
        memset(&mMeans[0],0,sizeof(Point3)*mK);

        for (size_t i=0; i<mClusters.size(); i++)
        {
            uint32_t id = mClusters[i];
            auto &mean = mMeans[id];
            counts[id]++;

            const auto &p = mData[i];
            mean.x+=p.x;
            mean.y+=p.y;
            mean.z+=p.z;
        }
        for (size_t i=0; i<mK; i++)
        {
            if ( counts[i] == 0 )
            {
                mMeans[i] = mOldMeans[i];
            }
            else
            {
                float recip = 1.0f / float(counts[i]);
                mMeans[i].x*=recip;
                mMeans[i].y*=recip;
                mMeans[i].z*=recip;
            }
        }
    }

    virtual void release(void) final
    {
        delete this;
    }

    void initializeClusters(void)
    {
        std::random_device rand_device;
        uint64_t seed = 0; //rand_device();
        // Using a very simple PRBS generator, parameters selected according to
        // https://en.wikipedia.org/wiki/Linear_congruential_generator#Parameters_in_common_use
        std::linear_congruential_engine<uint64_t, 6364136223846793005, 1442695040888963407, UINT64_MAX> rand_engine(seed);
        // Select first mean at random from the set
        {
            std::uniform_int_distribution<size_t> uniform_generator(0, mData.size() - 1);
            size_t rindex = uniform_generator(rand_engine);
            mMeans.push_back(mData[rindex]);
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

    DistanceVector closestDistance(const Point3Vector &means,
                                   const Point3Vector &data) 
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

    void calculateClusters(void)
    {
        for (size_t i=0; i<mData.size(); i++)
        {
            mClusters[i] = closestMean(mData[i]);
        }
    }

    uint32_t closestMean(const Point3 &p) const
    {
        uint32_t ret = 0;
        float closest = FLT_MAX;
        for (uint32_t i=0; i<mK; i++)
        {
            float d2 = p.distanceSquared(mMeans[i]);
            if ( d2 < closest )
            {
                closest = d2;
                ret = i;
            }
        }
        return ret;
    }

    uint32_t        mK{32};     // Maximum number of mean values to produce
    Point3Vector    mData;      // Input data
    Point3Vector    mMeans;     // Means
    Point3Vector    mOldMeans;  // Means on the previous iteration
    Point3Vector    mOldOldMeans;  // Means on the previous iteration
    ClusterVector   mClusters;  // Which cluster each source data point is in
};

Kmeans *Kmeans::create(void)
{
    auto ret = new KmeansImpl;
    return static_cast< Kmeans *>(ret);
}


}
#endif