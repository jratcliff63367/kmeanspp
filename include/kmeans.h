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

class KdTreeNode;

enum Axes
{
    X_AXIS = 0,
    Y_AXIS = 1,
    Z_AXIS = 2
};

class KdTreeFindNode
{
public:
    KdTreeFindNode() = default;

    KdTreeNode* m_node{ nullptr };
    float m_distance{ 0.0 };
};

/*
 * To minimize memory allocations while maintaining pointer stability.
 * Used in KdTreeNode and ConvexHull, as both use tree data structures that rely on pointer stability
 * Neither rely on random access or iteration
 * They just dump elements into a memory pool, then refer to pointers to the elements
 * All elements are default constructed in NodeStorage's m_nodes array
 */
template <typename T, std::size_t MaxBundleSize = 1024>
class NodeBundle
{
    struct NodeStorage {
        bool IsFull() const;

        T& GetNextNode();

        std::size_t m_index;
        std::array<T, MaxBundleSize> m_nodes;
    };

    std::list<NodeStorage> m_list;
    typename std::list<NodeStorage>::iterator m_head{ m_list.end() };

public:
    T& GetNextNode();

    T& GetFirstNode();

    void Clear();
};

template <typename T, std::size_t MaxBundleSize>
bool NodeBundle<T, MaxBundleSize>::NodeStorage::IsFull() const
{
    return m_index == MaxBundleSize;
}

template <typename T, std::size_t MaxBundleSize>
T& NodeBundle<T, MaxBundleSize>::NodeStorage::GetNextNode()
{
    assert(m_index < MaxBundleSize);
    T& ret = m_nodes[m_index];
    m_index++;
    return ret;
}

template <typename T, std::size_t MaxBundleSize>
T& NodeBundle<T, MaxBundleSize>::GetNextNode()
{
    /*
     * || short circuits, so doesn't dereference if m_bundle == m_bundleHead.end()
     */
    if (   m_head == m_list.end()
        || m_head->IsFull())
    {
        m_head = m_list.emplace(m_list.end());
    }

    return m_head->GetNextNode();
}

template <typename T, std::size_t MaxBundleSize>
T& NodeBundle<T, MaxBundleSize>::GetFirstNode()
{
    assert(m_head != m_list.end());
    return m_list.front().m_nodes[0];
}

template <typename T, std::size_t MaxBundleSize>
void NodeBundle<T, MaxBundleSize>::Clear()
{
    m_list.clear();
}

class Vertex
{
public:
    Vertex(void)
    {
    }
    Vertex(const Vertex &v) 
    {
        mPoint[0] = v.mPoint[0];
        mPoint[1] = v.mPoint[1];
        mPoint[2] = v.mPoint[2];
        mId = v.mId;
    }
    Vertex(float x,float y,float z,uint32_t id)
    {
        mPoint[0] = x;
        mPoint[1] = y;
        mPoint[2] = z;
        mId = id;
    }

    inline Vertex operator-(const Vertex &v) const
    {
        return Vertex( mPoint[0] - v.mPoint[0],
                       mPoint[1] - v.mPoint[1],
                       mPoint[2] - v.mPoint[2],mId);
    }

    float getNormSquared(void) const
    {
        return mPoint[0]*mPoint[0] + mPoint[1]*mPoint[1] + mPoint[2] *mPoint[2];
    }
    float   mPoint[3];
    uint32_t mId;
};

class KdTree
{
public:
    KdTree() = default;

    const Vertex& GetPosition(uint32_t index) const;

    uint32_t Search(const Vertex& pos,
                    float radius,
                    KdTreeFindNode &found) const;

    uint32_t Add(const Vertex& v);

    KdTreeNode& GetNewNode(uint32_t index);

    uint32_t GetNearest(const Vertex& pos,
                        float radius,
                        bool& _found) const; // returns the nearest possible neighbor's index.

    const std::vector<Vertex>& GetVertices() const;
    std::vector<Vertex>&& TakeVertices();

    uint32_t GetVCount() const;

private:
    KdTreeNode* m_root{ nullptr };
    NodeBundle<KdTreeNode> m_bundle;

    std::vector<Vertex> m_vertices;
};

class KdTreeNode
{
public:
    KdTreeNode() = default;
    KdTreeNode(uint32_t index);

    void Add(KdTreeNode& node,
             Axes dim,
             const KdTree& iface);

    uint32_t GetIndex() const;

    void Search(Axes axis,
                const Vertex& pos,
                float &radius,
                KdTreeFindNode &found,
                const KdTree& iface);

private:
    uint32_t m_index = 0;
    KdTreeNode* m_left = nullptr;
    KdTreeNode* m_right = nullptr;
};

const Vertex& KdTree::GetPosition(uint32_t index) const
{
    assert(index < m_vertices.size());
    return m_vertices[index];
}

uint32_t KdTree::Search(const Vertex& pos,
                        float _radius,
                        KdTreeFindNode &found) const
{
    if (!m_root)
        return 0;
    uint32_t count = 0;
    float radius = _radius;
    m_root->Search(X_AXIS, pos, radius, found, *this);
    return count;
}

uint32_t KdTree::Add(const Vertex& v)
{
    uint32_t ret = uint32_t(m_vertices.size());
    m_vertices.emplace_back(v);
    KdTreeNode& node = GetNewNode(ret);
    if (m_root)
    {
        m_root->Add(node,
                    X_AXIS,
                    *this);
    }
    else
    {
        m_root = &node;
    }
    return ret;
}

KdTreeNode& KdTree::GetNewNode(uint32_t index)
{
    KdTreeNode& node = m_bundle.GetNextNode();
    node = KdTreeNode(index);
    return node;
}

uint32_t KdTree::GetNearest(const Vertex& pos,
                            float radius,
                            bool& _found) const // returns the nearest possible neighbor's index.
{
    uint32_t ret = 0;

    _found = false;
    KdTreeFindNode found;
    found.m_distance = radius*radius;
    found.m_node = nullptr;
    uint32_t count = Search(pos, radius, found);
    if ( found.m_node)
    {
        KdTreeNode* node = found.m_node;
        ret = node->GetIndex();
        _found = true;
    }
    return ret;
}

const std::vector<Vertex>& KdTree::GetVertices() const
{
    return m_vertices;
}

std::vector<Vertex>&& KdTree::TakeVertices()
{
    return std::move(m_vertices);
}

uint32_t KdTree::GetVCount() const
{
    return uint32_t(m_vertices.size());
}

KdTreeNode::KdTreeNode(uint32_t index)
    : m_index(index)
{
}

void KdTreeNode::Add(KdTreeNode& node,
                     Axes dim,
                     const KdTree& tree)
{
    Axes axis = X_AXIS;
    uint32_t idx = 0;
    switch (dim)
    {
    case X_AXIS:
        idx = 0;
        axis = Y_AXIS;
        break;
    case Y_AXIS:
        idx = 1;
        axis = Z_AXIS;
        break;
    case Z_AXIS:
        idx = 2;
        axis = X_AXIS;
        break;
    }

    const Vertex& nodePosition = tree.GetPosition(node.m_index);
    const Vertex& position = tree.GetPosition(m_index);
    if (nodePosition.mPoint[idx] <= position.mPoint[idx])
    {
        if (m_left)
            m_left->Add(node, axis, tree);
        else
            m_left = &node;
    }
    else
    {
        if (m_right)
            m_right->Add(node, axis, tree);
        else
            m_right = &node;
    }
}

uint32_t KdTreeNode::GetIndex() const
{
    return m_index;
}

void KdTreeNode::Search(Axes axis,
                        const Vertex& pos,
                        float &radius,
                        KdTreeFindNode &found,
                        const KdTree& iface)
{
    // Get the position of this node
    const Vertex position = iface.GetPosition(m_index);
    // Compute the difference between this node position and the point
    // we are searching against
    const Vertex d = pos - position;

    KdTreeNode* search1 = 0;
    KdTreeNode* search2 = 0;

    // Compute the array index (X,Y,Z) and increment
    // the axis to the next search plane
    uint32_t idx = 0;
    switch (axis)
    {
        case X_AXIS:
            idx = 0;
            axis = Y_AXIS;
            break;
        case Y_AXIS:
            idx = 1;
            axis = Z_AXIS;
            break;
        case Z_AXIS:
            idx = 2;
            axis = X_AXIS;
            break;
    }

    if (d.mPoint[idx] <= 0) // JWR  if we are to the left
    {
        search1 = m_left; // JWR  then search to the left
        if (-d.mPoint[idx] < radius) // JWR  if distance to the right is less than our search radius, continue on the right
                            // as well.
            search2 = m_right;
    }
    else
    {
        search1 = m_right; // JWR  ok, we go down the left tree
        if (d.mPoint[idx] < radius) // JWR  if the distance from the right is less than our search radius
            search2 = m_left;
    }

    float r2 = radius * radius;
    float m = d.getNormSquared();
    // if the distance between this point and the radius match
    if (m < r2)
    {
        // If this is less than the current closest point found 
        // this becomes the new closest point found
        if (m < found.m_distance)
        {
            found.m_node = this;   // Remember the node
            found.m_distance = m;  // Remember the distance to this node
            radius = sqrtf(m);
        }
    }


    if (search1)
    {
        search1->Search(axis, pos, radius, found, iface);
    }

    if (search2)
    {
        search2->Search(axis, pos, radius, found, iface);
    }
}



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

    virtual const float *compute(const float *sourcePoints,
        uint32_t pointCount,
        uint32_t maxPoints,
        uint32_t &resultPointCount) final
    {
        const float *ret = nullptr;

        mK = maxPoints;
        mMeans.clear();
        mMeans.reserve(mK);
        // If the number of input points is less than or equal
        // to 'K' we just return the input points directly
        if ( pointCount <= mK )
        {
            mMeans.resize(pointCount);
            memcpy(&mMeans[0],sourcePoints,sizeof(Point3)*pointCount);
            resultPointCount = pointCount;
            ret = &mMeans[0].x;
            return ret;
        }

        mClusters.resize(pointCount);

        mData.resize(pointCount);
        memcpy(&mData[0],sourcePoints,sizeof(float)*3*pointCount);

        {
            Timer t;
            initializeClusters();
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
        DistanceVector distances;
        distances.resize(mData.size());
        for (uint32_t count = 1; count < mK; ++count) 
        {
            Timer t;
            // Calculate the distance to the closest mean for each data point
            closestDistance(mMeans, mData, distances);
            mTimeClosestDistances+=t.getElapsedSeconds();
            // Pick a random point weighted by the distance from existing means
            // TODO: This might convert floating point weights to ints, distorting the distribution for small weights
            std::discrete_distribution<size_t> generator(distances.begin(), distances.end());
            mMeans.push_back(mData[generator(rand_engine)]);
            mTimeRandomSampling+=t.getElapsedSeconds();
        }
    }

    void closestDistance(const Point3Vector &means,const Point3Vector &data,DistanceVector &distances) 
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
    double           mLimitDelta{0.001f};
    double           mTimeInitializing{0};
    double           mTimeClusters{0};
    double           mTimeMeans{0};
    double           mTimeTermination{0};
    double           mTimeClosestDistances{0};
    double           mTimeRandomSampling{0};
};

Kmeans *Kmeans::create(void)
{
    auto ret = new KmeansImpl;
    return static_cast< Kmeans *>(ret);
}


}
#endif