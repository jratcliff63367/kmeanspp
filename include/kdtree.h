#pragma once

namespace kdtree
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
    struct NodeStorage 
    {
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
    if (   m_head == m_list.end() || m_head->IsFull())
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

    // If the distance from the position we are searching against
    // and the root node is less than the search radius provided then
    // we shrink the search radius down since the root node is already
    // the 'nearest' relative to the search criteria given
    const Vertex &rootPos = GetPosition(m_root->GetIndex());
    float d2 = rootPos.getDistanceSquared(pos);
    float pdist = sqrtf(d2);
    if ( pdist < radius )
    {
        radius = pdist;
    }
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

}
