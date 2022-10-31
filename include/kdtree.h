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
        bool isFull() const
        {
            return m_index == MaxBundleSize;
        }

        T& getNextNode()
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
    T& getNextNode()
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

    T& getFirstNode()
    {
        assert(m_head != m_list.end());
        return m_list.front().m_nodes[0];
    }

    void clear()
    {
        m_list.clear();
    }
};


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

    float getDistanceSquared(const Vertex &v) const
    {
        float dx = v.mPoint[0] - mPoint[0];
        float dy = v.mPoint[1] - mPoint[1];
        float dz = v.mPoint[2] - mPoint[2];
        return dx*dx + dy*dy + dz*dz;
    }

    float   mPoint[3];
    uint32_t mId;
};

class KdTreeInterface
{
public:
    virtual const Vertex& getPosition(uint32_t index) const = 0;
};

class KdTreeNode;

// This is a small struct used internally when doing a search of 
//the KdTree. It returns which node and what distance matched the
// search criteria
class KdTreeFindNode
{
public:
    KdTreeFindNode() = default;

    KdTreeNode* m_node{ nullptr };
    float m_distance{ 0.0 };
};

class KdTreeNode
{
public:
    KdTreeNode() = default;
    KdTreeNode(uint32_t index) : m_index(index)
    {
    }

    void add(KdTreeNode& node,
        Axes dim,
        const KdTreeInterface& tree)
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

        const Vertex& nodePosition = tree.getPosition(node.m_index);
        const Vertex& position = tree.getPosition(m_index);
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

    uint32_t getIndex() const
    {
        return m_index;
    }

    void search(Axes axis,
        const Vertex& pos,
        float &radius,
        KdTreeFindNode &found,
        const KdTreeInterface& iface)
    {
        // If we have found something with a distance of zero we can stop searching
        if ( found.m_node && found.m_distance == 0 )
        {
            return;
        }
        // Get the position of this node
        const Vertex position = iface.getPosition(m_index);
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
                // If we have found something with a distance of zero we can stop searching
                if (found.m_node && found.m_distance == 0)
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
    KdTreeNode* m_left = nullptr;
    KdTreeNode* m_right = nullptr;
};




class KdTree : public KdTreeInterface
{
public:
    KdTree() = default;

    virtual const Vertex& getPosition(uint32_t index) const
    {
        assert(index < m_vertices.size());
        return m_vertices[index];
    }

    // Search for the nearest position within the radius provided
    // If the return value is -1 that means we could not find a point in range
    // If the value is greater than or equal to zero then that is the distance
    // between the search position and the nearest position.
    // The result position and index is stored in 'result'.
    float search(const Vertex& pos,
                    float _radius,
                    Vertex &result) const
    {
        float ret = -1;
        if (!m_root)
            return ret;
        float radius = _radius;
        KdTreeFindNode found;
        // If the distance from the position we are searching against
        // and the root node is less than the search radius provided then
        // we shrink the search radius down since the root node is already
        // the 'nearest' relative to the search criteria given
        const Vertex &rootPos = getPosition(m_root->getIndex());
        float d2 = rootPos.getDistanceSquared(pos);
        float pdist = sqrtf(d2);
        if (pdist <= radius)
        {
            radius = pdist;
            found.m_distance = pdist;
            found.m_node = m_root;
        }
        m_root->search(X_AXIS, pos, radius, found, *this);
        if ( found.m_node )
        {
            ret = found.m_distance;
            result = getPosition(found.m_node->getIndex());
        }
        return ret;
    }

    uint32_t add(const Vertex& v)
    {
        uint32_t ret = uint32_t(m_vertices.size());
        m_vertices.emplace_back(v);
        KdTreeNode& node = getNewNode(ret);
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

    KdTreeNode& getNewNode(uint32_t index)
    {
        KdTreeNode& node = m_bundle.getNextNode();
        node = KdTreeNode(index);
        return node;
    }

    const std::vector<Vertex>& getVertices() const
    {
        return m_vertices;
    }

    std::vector<Vertex>&& takeVertices()
    {
        return std::move(m_vertices);
    }

    uint32_t getVCount() const
    {
        return uint32_t(m_vertices.size());
    }

private:
    KdTreeNode* m_root{ nullptr };
    NodeBundle<KdTreeNode> m_bundle;

    std::vector<Vertex> m_vertices;
};


}
