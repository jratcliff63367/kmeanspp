#pragma once

#include <stdint.h>

#define USE_KDTREE 0

#if USE_KDTREE
#include "kdtree.h"
#endif

namespace kmeans
{


class Kmeans
{
public:
    class Parameters
    {
    public:
        const float *mPoints{nullptr};
        uint32_t     mPointCount{0};
        uint32_t     mMaxPoints{0}; // maximum number of output points
        // If this is the same size as max-points then we just
        // do a random distribution and bypass the kmeans++ computation
        uint32_t     mMaximumPlusPlusCount{0};
    };
    static Kmeans *create(void);

    virtual const float *compute(const Parameters &params,uint32_t &resultPointCount) = 0;

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
#include <list>

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
#include <chrono>

namespace kmeans
{

class RandPool
{
public:
    RandPool(uint32_t size) // size of random number bool.
    {
        mData = new uint32_t[size];
        mSize = size;
        mTop = mSize;
        for (uint32_t i = 0; i < mSize; i++)
        {
            mData[i] = i;
        }
    }

    ~RandPool(void)
    {
        delete []mData;
    };

    // pull a number from the random number pool, will never return the
    // same number twice until the 'deck' (pool) has been exhausted.
    // Will set the shuffled flag to true if the deck/pool was exhausted
    // on this call.
    uint32_t get(bool& shuffled)
    {
        if (mTop == 0) // deck exhausted, shuffle deck.
        {
            shuffled = true;
            mTop = mSize;
        }
        else
        {
            shuffled = false;
        }
        uint32_t entry = rand() % mTop;
        mTop--;
        uint32_t ret = mData[entry]; // swap top of pool with entry
        mData[entry] = mData[mTop]; // returned
        mData[mTop] = ret;
        return ret;
    };


private:
    uint32_t* mData; // random number bool.
    uint32_t mSize; // size of random number pool.
    uint32_t mTop; // current top of the random number pool.
};

    class Timer
    {
    public:
        Timer() : mStartTime(std::chrono::high_resolution_clock::now())
        {
        }

        void reset()
        {
            mStartTime = std::chrono::high_resolution_clock::now();
        }

        double getElapsedSeconds()
        {
            auto s = peekElapsedSeconds();
            reset();
            return s;
        }

        double peekElapsedSeconds()
        {
            auto now = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> diff = now - mStartTime;
            return diff.count();
        }

    private:
        std::chrono::time_point<std::chrono::high_resolution_clock> mStartTime;
    };


class Point3
{
public:

    bool operator!=(const Point3 &p) const
    {
        bool ret = false;

        if ( p.x != x || p.y != y || p.z != z )
        {
            ret = true;
        }

        return ret;
    }

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

    virtual const float *compute(const Parameters &params,
        uint32_t &resultPointCount) final
    {
        const float *ret = nullptr;

        mK = params.mMaxPoints;
        mMeans.clear();
        mMeans.reserve(mK);
        // If the number of input points is less than or equal
        // to 'K' we just return the input points directly
        if ( params.mPointCount <= mK )
        {
            mMeans.resize(params.mPointCount);
            memcpy(&mMeans[0],params.mPoints,sizeof(Point3)*params.mPointCount);
            resultPointCount = params.mPointCount;
            ret = &mMeans[0].x;
            return ret;
        }

        mClusters.resize(params.mPointCount);
        mData.resize(params.mPointCount);
        memcpy(&mData[0],params.mPoints,sizeof(float)*3*params.mPointCount);

        {
            Timer t;
            initializeClusters(params);
            mTimeInitializing = t.getElapsedSeconds();
        }

        uint32_t count = 0;
        uint32_t maxIterations = 100;
        mOldMeans = mMeans;
        mOldOldMeans = mMeans;
        do 
        {
            {
                Timer t;
                calculateClusters();
                mTimeClusters+=t.getElapsedSeconds();
            }
            Timer t;
            mOldOldMeans = mOldMeans;
            mOldMeans = mMeans;
            calculateMeans();
            mTimeMeans+=t.getElapsedSeconds();
            count++;
            Timer tm;
            if ( sameMeans(mMeans,mOldMeans))
            {
                break;
            }
            if (sameMeans(mMeans, mOldOldMeans))
            {
                break;
            }
            mTimeTermination+=tm.getElapsedSeconds();
        } while ( count < maxIterations );

        resultPointCount = mK;
        ret = &mMeans[0].x;

        printf("Ran             : %d iterations.\n",count);
        printf("TimeInitializing: %0.2f seconds\n",mTimeInitializing);
        printf("ClosestDistances: %0.2f seconds\n",mTimeClosestDistances);
        printf("RandomSampling:   %0.2f seconds\n",mTimeRandomSampling);
        printf("BuildingKdTree:   %0.2f seconds\n",mTimeRebuildingKdTree);
        printf("TimeClusters:     %0.2f seconds\n",mTimeClusters);
        printf("TimeMeans:        %0.2f seconds\n",mTimeMeans);
        printf("TimeTermination:  %0.2f seconds\n",mTimeTermination);

        return ret;
    }

    bool nearlySameMeans(const Point3Vector &a, const Point3Vector &b)
    {
        bool ret = true;

        for (size_t i=0; i<a.size(); i++)
        {
            double d = a[i].distanceSquared(b[i]);
            if ( d > mLimitDelta )
            {
                ret = false;
            }
        }

        return ret;
    }


    bool sameMeans(const Point3Vector &a,const Point3Vector &b)
    {
        bool ret = true;

        for (size_t i=0; i<a.size(); i++)
        {
            if ( a[i] != b[i] )
            {
                ret = false;
                break;
            }
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

    void initializeClusters(const Parameters &params)
    {
        uint32_t maxPlusPlusCount = params.mMaximumPlusPlusCount;
        if ( maxPlusPlusCount < params.mMaxPoints )
        {
            maxPlusPlusCount = params.mMaxPoints;
        }
        else if ( maxPlusPlusCount > params.mPointCount )
        {
            maxPlusPlusCount = params.mPointCount;
        }
        if ( params.mMaxPoints == maxPlusPlusCount )
        {
            mMeans.clear();
            mMeans.resize(maxPlusPlusCount);
            RandPool rp(params.mPointCount);
            for (uint32_t i=0; i<maxPlusPlusCount; i++)
            {
                bool shuffled;
                uint32_t index = rp.get(shuffled);
                const auto &p = mData[index];
                mMeans[i] = p;
            }
            return;
        }
        uint32_t dataSize = uint32_t(mData.size());
        Point3Vector data;
        if ( maxPlusPlusCount != dataSize )
        {
            data.resize(maxPlusPlusCount);
            RandPool rp(dataSize);
            for (uint32_t i=0; i<maxPlusPlusCount; i++)
            {
                bool shuffled;
                uint32_t index = rp.get(shuffled);
                data[i] = mData[index];
            }
        }
        else
        {
            data = mData;
        }
        std::random_device rand_device;
        uint64_t seed = 0; //rand_device();
        // Using a very simple PRBS generator, parameters selected according to
        // https://en.wikipedia.org/wiki/Linear_congruential_generator#Parameters_in_common_use
        std::linear_congruential_engine<uint64_t, 6364136223846793005, 1442695040888963407, UINT64_MAX> rand_engine(seed);
        // Select first mean at random from the set
        {
            std::uniform_int_distribution<size_t> uniform_generator(0, data.size() - 1);
            size_t rindex = uniform_generator(rand_engine);
            mMeans.push_back(data[rindex]);
        }
#if USE_KDTREE
        kdtree::KdTree kdt;
        kdt.reservePoints(mK);
        kdtree::KdPoint p;
        const auto &m = mMeans[0];
        p.mId = 0;
        p.mPos[0] = m.x;
        p.mPos[1] = m.y;
        p.mPos[2] = m.z;
        kdt.addPoint(p);
        kdt.buildTree();
#endif
        DistanceVector distances;
        distances.resize(data.size());
        for (uint32_t count = 1; count < mK; ++count) 
        {
            Timer t;
            // Calculate the distance to the closest mean for each data point
#if USE_KDTREE
            closestDistance(kdt, data, distances);
#else
            closestDistance(mMeans, data, distances);
#endif
            mTimeClosestDistances+=t.getElapsedSeconds();
            // Pick a random point weighted by the distance from existing means
            // TODO: This might convert floating point weights to ints, distorting the distribution for small weights
            std::discrete_distribution<size_t> generator(distances.begin(), distances.end());
            uint32_t index = (uint32_t)mMeans.size();
            mMeans.push_back(data[generator(rand_engine)]);
#if USE_KDTREE
            kdtree::KdPoint p;
            const auto &m = mMeans[index];
            p.mId = index;
            p.mPos[0] = m.x;
            p.mPos[1] = m.y;
            p.mPos[2] = m.z;
            kdt.addPoint(p);
            Timer tt;
            kdt.buildTree();
            mTimeRebuildingKdTree+=tt.getElapsedSeconds();
#endif
            mTimeRandomSampling+=t.getElapsedSeconds();
        }
    }

    void closestDistance(const Point3Vector &means,
                         const Point3Vector &data,
                         DistanceVector &distances) 
    {
        uint32_t index = 0;
        for (auto& d : data) 
        {
            float closest = FLT_MAX;
            for (auto& m : means) 
            {
                float distance = d.distanceSquared(m);
                if (distance < closest)
                {
                    closest = distance;
                }
            }
            distances[index] = closest;
            index++;
        }
    }

#if USE_KDTREE
    void closestDistance(kdtree::KdTree &kdt,
        const Point3Vector &data,
        DistanceVector &distances)
    {
        uint32_t index = 0;
        for (auto& d : data)
        {
            kdtree::KdPoint p(d.x,d.y,d.z),r;
            distances[index] = kdt.findNearest(p,r);
            index++;
        }
    }
#endif

    void calculateClusters(void)
    {
#if USE_KDTREE
        kdtree::KdTree kdt;
        uint32_t msize = uint32_t(mMeans.size());
        kdt.reservePoints(msize);
        for (uint32_t i=0; i<msize; i++)
        {
            const auto &p = mMeans[i];
            kdtree::KdPoint kp;
            kp.mPos[0] = p.x;
            kp.mPos[1] = p.y;
            kp.mPos[2] = p.z;
            kp.mId = i;
            kdt.addPoint(kp);
        }
        kdt.buildTree();
        for (size_t i = 0; i < mData.size(); i++)
        {
            const auto &p = mData[i];
            kdtree::KdPoint kp;
            kp.mPos[0] = p.x;
            kp.mPos[1] = p.y;
            kp.mPos[2] = p.z;
            kdtree::KdPoint result;
            kdt.findNearest(kp,result);
            mClusters[i] = result.mId;
        }
#else
        for (size_t i=0; i<mData.size(); i++)
        {
            mClusters[i] = closestMean(mData[i]);
        }
#endif
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
    double           mLimitDelta{0.001f};
    double           mTimeInitializing{0};
    double           mTimeClusters{0};
    double           mTimeMeans{0};
    double           mTimeTermination{0};
    double           mTimeClosestDistances{0};
    double           mTimeRandomSampling{0};
    double           mTimeRebuildingKdTree{0};
};

Kmeans *Kmeans::create(void)
{
    auto ret = new KmeansImpl;
    return static_cast< Kmeans *>(ret);
}


}
#endif
