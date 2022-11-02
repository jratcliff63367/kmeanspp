#pragma once

#include <stdint.h>

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
        uint32_t     mMaxIterations{100};
        bool        mUseKdTree{true}; // you would always want this to true unless trying to debug something
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

#include "kdtree.h"

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
        // Resize the means container to have room for
        // a total of three copies.
        // When iterating on kmeans we stop if the means results are
        // the same as the previous iteration *or* the one before that.
        // To avoid memory copying we instead just cycle pointers
        size_t msize = mMeans.size();
        mMeans.resize(msize*3);
        mCurrentMeans = &mMeans[0];
        mOldMeans = &mMeans[msize];
        mOldOldMeans = &mMeans[msize*2];
        memcpy(mOldMeans,mCurrentMeans,sizeof(Point3)*msize);
        memcpy(mOldOldMeans,mCurrentMeans,sizeof(Point3)*msize);

        do 
        {
            {
                Timer t;
                calculateClusters(params.mUseKdTree,msize);
                mTimeClusters+=t.getElapsedSeconds();
            }
            Timer t;
            // Pointer swap, the current means is now the old means.
            // The old means is now the old-old means
            // And the old old means pointer now becomes the current means pointer
            Point3 *temp = mOldOldMeans;
            mOldOldMeans = mOldMeans;
            mOldMeans = mCurrentMeans;
            mCurrentMeans = temp;

            calculateMeans(mCurrentMeans,msize,mOldMeans);

            mTimeMeans+=t.getElapsedSeconds();
            count++;
            Timer tm;
            if ( sameMeans(mCurrentMeans,mOldMeans,msize))
            {
                break;
            }
            if (sameMeans(mCurrentMeans, mOldOldMeans,msize))
            {
                break;
            }
            mTimeTermination+=tm.getElapsedSeconds();
        } while ( count < params.mMaxIterations );

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


    bool sameMeans(const Point3 *a,const Point3 *b,size_t msize)
    {
        bool ret = true;

        for (size_t i=0; i<msize; i++)
        {
            if ( a[i] != b[i] )
            {
                ret = false;
                break;
            }
        }

        return ret;
    }

    void calculateMeans(Point3 *means,
                        size_t msize,
                        const Point3 *oldMeans)
    {
        std::vector< uint32_t > counts;
        assert( mData.size() == mClusters.size() );
        counts.resize(msize);
        memset(&counts[0],0,sizeof(uint32_t)*msize);
        memset(means,0,sizeof(Point3)*msize);

        for (size_t i=0; i<mClusters.size(); i++)
        {
            uint32_t id = mClusters[i];
            assert( id < uint32_t(msize) );
            auto &mean = means[id];
            counts[id]++;

            const auto &p = mData[i];
            mean.x+=p.x;
            mean.y+=p.y;
            mean.z+=p.z;
        }
        for (size_t i=0; i<msize; i++)
        {
            if ( counts[i] == 0 )
            {
                means[i] = oldMeans[i];
            }
            else
            {
                float recip = 1.0f / float(counts[i]);
                means[i].x*=recip;
                means[i].y*=recip;
                means[i].z*=recip;
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
        kdtree::KdTree kdt;
        if ( params.mUseKdTree )
        {
            kdt.reservePoints(mK);
            kdtree::KdPoint p;
            const auto &m = mMeans[0];
            p.mId = 0;
            p.mPos[0] = m.x;
            p.mPos[1] = m.y;
            p.mPos[2] = m.z;
            kdt.addPoint(p);
            kdt.buildTree();
        }

        DistanceVector distances;
        distances.resize(data.size());
        for (uint32_t count = 1; count < mK; ++count) 
        {
            Timer t;
            // Calculate the distance to the closest mean for each data point
            if ( params.mUseKdTree )
            {
                closestDistance(kdt, data, distances);
            }
            else
            {
                closestDistance(mMeans, data, distances);
            }
            mTimeClosestDistances+=t.getElapsedSeconds();
            // Pick a random point weighted by the distance from existing means
            // TODO: This might convert floating point weights to ints, distorting the distribution for small weights
            std::discrete_distribution<size_t> generator(distances.begin(), distances.end());
            uint32_t index = (uint32_t)mMeans.size();
            mMeans.push_back(data[generator(rand_engine)]);
            if ( params.mUseKdTree )
            {
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
            }
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

    void calculateClusters(bool useKdTree,size_t msize)
    {
        if ( useKdTree )
        {
            kdtree::KdTree kdt;
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
                assert( result.mId < msize );
                mClusters[i] = result.mId;
            }
        }
        else
        {
            for (size_t i=0; i<mData.size(); i++)
            {
                mClusters[i] = closestMean(mData[i]);
            }
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

    Point3          *mCurrentMeans{nullptr};
    Point3          *mOldMeans{nullptr};
    Point3          *mOldOldMeans{nullptr};

    Point3Vector    mMeans;     // Means

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
