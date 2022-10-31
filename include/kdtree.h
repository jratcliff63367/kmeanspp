#pragma once

namespace kdtree
{

enum Axes
{
    X_AXIS = 0,
    Y_AXIS = 1,
    Z_AXIS = 2
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
    struct NodeStorage 
    {
        inline bool isFull() const
        {
            return m_index == MaxBundleSize;
        }

        inline T& getNextNode()
        {
            assert(m_index < MaxBundleSize);
            T& ret = m_nodes[m_index];
            m_index++;
            return ret;
        }

        std::size_t m_index;
        std::array<T, MaxBundleSize> m_nodes;
    };

    std::list<NodeStorage> m_list;
    typename std::list<NodeStorage>::iterator m_head{ m_list.end() };

public:
    inline T& getNextNode()
    {
        /*
         * || short circuits, so doesn't dereference if m_bundle == m_bundleHead.end()
         */
        if (   m_head == m_list.end() || m_head->isFull())
        {
            m_head = m_list.emplace(m_list.end());
        }
        return m_head->getNextNode();
    }

    inline T& getFirstNode()
    {
        assert(m_head != m_list.end());
        return m_list.front().m_nodes[0];
    }

    inline void clear()
    {
        m_list.clear();
    }
};


template <class Type> class Vertex
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
    Vertex(Type x,Type y,Type z,uint32_t id)
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

    inline Type getNormSquared(void) const
    {
        return mPoint[0]*mPoint[0] + mPoint[1]*mPoint[1] + mPoint[2] *mPoint[2];
    }

    inline Type getDistanceSquared(const Vertex &v) const
    {
        Type dx = v.mPoint[0] - mPoint[0];
        Type dy = v.mPoint[1] - mPoint[1];
        Type dz = v.mPoint[2] - mPoint[2];
        return dx*dx + dy*dy + dz*dz;
    }

    Type   mPoint[3];
    uint32_t mId;
};

template <class Type>class KdTreeInterface
{
public:
    virtual const Vertex<Type>& getPosition(uint32_t index) const = 0;
};

template <class Type> class KdTreeNode;

// This is a small struct used internally when doing a search of 
//the KdTree. It returns which node and what distance matched the
// search criteria
template <class Type> class KdTreeFindNode
{
public:
    KdTreeFindNode() = default;

    KdTreeNode<Type>* mNode{ nullptr };
    Type mDistanceSquared{ 0.0 };
    uint32_t    mTestCount{0};
};

template <class Type> class KdTreeNode
{
public:
    KdTreeNode<Type>() = default;
    KdTreeNode<Type>(uint32_t index) : m_index(index)
    {
    }

    inline void add(KdTreeNode<Type>& node,
        Axes dim,
        const KdTreeInterface<Type>& tree)
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

        const Vertex<Type>& nodePosition = tree.getPosition(node.m_index);
        const Vertex<Type>& position = tree.getPosition(m_index);
        if (nodePosition.mPoint[idx] <= position.mPoint[idx])
        {
            if (m_left)
                m_left->add(node, axis, tree);
            else
                m_left = &node;
        }
        else
        {
            if (m_right)
                m_right->add(node, axis, tree);
            else
                m_right = &node;
        }
    }

    inline uint32_t getIndex() const
    {
        return m_index;
    }

    inline void search(Axes axis,
        const Vertex<Type>& pos,
        Type &radius,
        KdTreeFindNode<Type> &found,
        const KdTreeInterface<Type>& iface)
    {
        // If we have found something with a distance of zero we can stop searching
        if ( found.mNode && found.mDistanceSquared == 0 )
        {
            return;
        }
        // Get the position of this node
        const Vertex<Type> position = iface.getPosition(m_index);
        // Compute the difference between this node position and the point
        // we are searching against
        const Vertex<Type> d = pos - position;

        KdTreeNode<Type>* search1 = 0;
        KdTreeNode<Type>* search2 = 0;

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
        found.mTestCount++;
        Type r2 = radius * radius;
        Type m = position.getDistanceSquared(pos);
        // if the distance between this point and the radius match
        if (m < r2)
        {
            // If this is less than the current closest point found
            // this becomes the new closest point found
            if (m < found.mDistanceSquared)
            {
                found.mNode = this;   // Remember the node
                found.mDistanceSquared = m;  // Remember the distance to this node
                radius = (Type)sqrt(m);
                // If we have found something with a distance of zero we can stop searching
                if (found.mNode && found.mDistanceSquared == 0)
                {
                    search1 = nullptr;
                    search2 = nullptr;
                }
            }
        }


        if (search1)
        {
            search1->search(axis, pos, radius, found, iface);
        }

        if (search2)
        {
            search2->search(axis, pos, radius, found, iface);
        }
    }

private:
    uint32_t m_index = 0;
    KdTreeNode<Type>* m_left = nullptr;
    KdTreeNode<Type>* m_right = nullptr;
};




template <class Type> class KdTree : public KdTreeInterface<Type>
{
public:
    KdTree() = default;

    virtual const Vertex<Type>& getPosition(uint32_t index) const
    {
        assert(index < m_vertices.size());
        return m_vertices[index];
    }

    // Search for the nearest position within the radius provided
    // If the return value is -1 that means we could not find a point in range
    // If the value is greater than or equal to zero then that is the distance
    // between the search position and the nearest position.
    // The result position and index is stored in 'result'.
    Type findNearest(const Vertex<Type>& pos,
                    Type _radius,
                    uint32_t &searchCount,
                    Vertex<Type> &result) const
    {
        Type ret = -1;
        searchCount = 0;
        if (!m_root)
            return ret;
        Type radius = _radius;
        KdTreeFindNode<Type> found;
        // If the distance from the position we are searching against
        // and the root node is less than the search radius provided then
        // we shrink the search radius down since the root node is already
        // the 'nearest' relative to the search criteria given
        const Vertex<Type> &rootPos = getPosition(m_root->getIndex());
        Type d2 = rootPos.getDistanceSquared(pos);
        Type pdist = (Type)sqrt(d2);
        if (pdist <= radius)
        {
            radius = pdist;
            found.mDistanceSquared = radius*radius;
            found.mNode = m_root;
        }
        m_root->search(X_AXIS, pos, radius, found, *this);
        if ( found.mNode )
        {
            ret = (Type) sqrt(found.mDistanceSquared);
            result = getPosition(found.mNode->getIndex());
            searchCount = found.mTestCount;
        }
        return ret;
    }

    inline uint32_t add(const Vertex<Type>& v)
    {
        uint32_t ret = uint32_t(m_vertices.size());
        m_vertices.emplace_back(v);
        KdTreeNode<Type>& node = getNewNode(ret);
        if (m_root)
        {
            m_root->add(node,
                X_AXIS,
                *this);
        }
        else
        {
            m_root = &node;
        }
        return ret;
    }

    inline KdTreeNode<Type>& getNewNode(uint32_t index)
    {
        KdTreeNode<Type>& node = m_bundle.getNextNode();
        node = KdTreeNode<Type>(index);
        return node;
    }

    inline const std::vector<Vertex<Type>>& getVertices() const
    {
        return m_vertices;
    }

    inline std::vector<Vertex<Type>>&& takeVertices()
    {
        return std::move(m_vertices);
    }

    inline uint32_t getVCount() const
    {
        return uint32_t(m_vertices.size());
    }

private:
    KdTreeNode<Type>* m_root{ nullptr };
    NodeBundle<KdTreeNode<Type>> m_bundle;

    std::vector<Vertex<Type>> m_vertices;
};


}
