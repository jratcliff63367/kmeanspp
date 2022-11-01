#pragma once

#include <stdint.h>

namespace kdtree
{

class KdPoint
{
public:
    uint32_t      mId{ 0 };
    float         mPos[3]{ 0,0,0 };
};

class KdNode
{
public:
    KdPoint       mPoint;
    KdNode       *mLeft{nullptr};
    KdNode       *mRight{nullptr};
};


class KdTree
{
public:
    KdTree(void)
    {
    }

    ~KdTree(void)
    {
    }

    inline void swap(KdNode *a,KdNode *b) 
    {
        KdNode temp;
        temp.mPoint = a->mPoint;
        a->mPoint = b->mPoint;
        b->mPoint = temp.mPoint;
    }

    KdNode *findMedian(KdNode *start,KdNode *end,int idx)
    {
        if (end <= start) return nullptr;
        if (end == start + 1)
        {
            return start;
        }

        KdNode *p, *store, *md = start + (end - start) / 2;
        float pivot;
        while (1) 
        {
            pivot = md->mPoint.mPos[idx];

            swap(md, end - 1);
            for (store = p = start; p < end; p++) 
            {
                if (p->mPoint.mPos[idx] < pivot) 
                {
                    if (p != store)
                    {
                        swap(p, store);
                    }
                    store++;
                }
            }
            swap(store, end - 1);

            /* median has duplicate values */
            if (store->mPoint.mPos[idx] == md->mPoint.mPos[idx])
            {
                break;
            }

            if (store > md) 
            {
                end = store;
            }
            else
            {
                start = store;
            }
        }
        return md;
    }

    KdNode *buildTree(KdNode *nodes,uint32_t nodeCount,uint32_t index)
    {
        KdNode *n;
        if ( nodeCount == 0 ) return nullptr;
        if ( ( n = findMedian(nodes,nodes+nodeCount,index)))
        {
            index = (index+1)%3;
            n->mLeft = buildTree(nodes,uint32_t(n-nodes),index);
            n->mRight = buildTree(n+1,uint32_t(nodes+nodeCount - (n+1)), index);
        }
        return n;
    }

    float dist(const KdNode *a,const KdNode *b)
    {
        float dx = a->mPoint.mPos[0] - b->mPoint.mPos[0];
        float dy = a->mPoint.mPos[1] - b->mPoint.mPos[1];
        float dz = a->mPoint.mPos[2] - b->mPoint.mPos[2];
        return dx*dx + dy*dy + dz*dz;
    }

    void nearest(const KdNode *root,
                 const KdNode *nd,
                 int index,
                 const KdNode *&best,
                 float &nearestDistanceSquared)
    {
        float d, dx, dx2;

        if (!root) return;

        d = dist(root, nd);
        dx = root->mPoint.mPos[index] - nd->mPoint.mPos[index];
        dx2 = dx * dx;

        if (!best || d < nearestDistanceSquared) 
        {
            nearestDistanceSquared = d;
            best = root;
        }

        /* if chance of exact match is high */
        if (nearestDistanceSquared == 0) return;

        index = (index+1)%3;

        nearest(dx > 0 ? root->mLeft : root->mRight, nd, index, best, nearestDistanceSquared);
        if (dx2 >= nearestDistanceSquared) return;
        nearest(dx > 0 ? root->mRight : root->mLeft, nd, index, best, nearestDistanceSquared);

    }

private:
};


}
