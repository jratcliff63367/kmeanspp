#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <random>

#include "ScopedTime.h"
#define ENABLE_KMEANS_IMPLEMENTATION 1
#include "kmeanspp.h"
#include "kdtree.h"

#define MAX_DIM 3
struct kd_node_t{
    double x[MAX_DIM];
    struct kd_node_t *left, *right;
};

    inline double
dist(struct kd_node_t *a, struct kd_node_t *b, int dim)
{
    double t, d = 0;
    while (dim--) {
        t = a->x[dim] - b->x[dim];
        d += t * t;
    }
    return d;
}
inline void swap(struct kd_node_t *x, struct kd_node_t *y) {
    double tmp[MAX_DIM];
    memcpy(tmp,  x->x, sizeof(tmp));
    memcpy(x->x, y->x, sizeof(tmp));
    memcpy(y->x, tmp,  sizeof(tmp));
}


/* see quickselect method */
    struct kd_node_t*
find_median(struct kd_node_t *start, struct kd_node_t *end, int idx)
{
    if (end <= start) return NULL;
    if (end == start + 1)
        return start;

    struct kd_node_t *p, *store, *md = start + (end - start) / 2;
    double pivot;
    while (1) {
        pivot = md->x[idx];

        swap(md, end - 1);
        for (store = p = start; p < end; p++) {
            if (p->x[idx] < pivot) {
                if (p != store)
                    swap(p, store);
                store++;
            }
        }
        swap(store, end - 1);

        /* median has duplicate values */
        if (store->x[idx] == md->x[idx])
            return md;

        if (store > md) end = store;
        else        start = store;
    }
}

    struct kd_node_t*
make_tree(struct kd_node_t *t, int len, int i, int dim)
{
    struct kd_node_t *n;

    if (!len) return 0;

    if ((n = find_median(t, t + len, i))) {
        i = (i + 1) % dim;
        n->left  = make_tree(t,int(n - t), i, dim);
        n->right = make_tree(n + 1, int(t + len - (n + 1)), i, dim);
    }
    return n;
}

/* global variable, so sue me */
int visited;

void nearest(struct kd_node_t *root, struct kd_node_t *nd, int i, int dim,
        struct kd_node_t **best, double *best_dist)
{
    double d, dx, dx2;

    if (!root) return;
    d = dist(root, nd, dim);
    dx = root->x[i] - nd->x[i];
    dx2 = dx * dx;

    visited ++;

    if (!*best || d < *best_dist) {
        *best_dist = d;
        *best = root;
    }

    /* if chance of exact match is high */
    if (!*best_dist) return;

    if (++i >= dim) i = 0;

    nearest(dx > 0 ? root->left : root->right, nd, i, dim, best, best_dist);
    if (dx2 >= *best_dist) return;
    nearest(dx > 0 ? root->right : root->left, nd, i, dim, best, best_dist);
}

#define N 1000000
#define rand1() (rand() / (double)RAND_MAX)
#define rand_pt(v) { v.x[0] = rand1(); v.x[1] = rand1(); v.x[2] = rand1(); }
int testKdTree(void)
{
    int i;
    struct kd_node_t wp[] = {
        {{2, 3,0}}, {{5, 4,0}}, {{9, 6,0}}, {{4, 7,0}}, {{8, 1,0}}, {{7, 2,0}}
    };
    struct kd_node_t testNode = {{9, 2,0}};
    struct kd_node_t *root, *found, *million;
    double best_dist;

    root = make_tree(wp, sizeof(wp) / sizeof(wp[1]), 0, 3);

    {
        kdtree::KdNode *nodes = new kdtree::KdNode[6];
        for (uint32_t i=0; i<6; i++)
        {
            nodes[i].mPoint.mPos[0] = (float) wp[i].x[0];
            nodes[i].mPoint.mPos[1] = (float)wp[i].x[1];
            nodes[i].mPoint.mPos[2] = (float)wp[i].x[2];
            nodes[i].mPoint.mId = i;
        }
        kdtree::KdTree kt;
        kdtree::KdNode *kroot = kt.buildTree(nodes,6,0);

        const kdtree::KdNode *kfound = nullptr;
        float bestDist = FLT_MAX;
        kdtree::KdNode ktestNode;
        ktestNode.mPoint.mPos[0] = (float)testNode.x[0];
        ktestNode.mPoint.mPos[1] = (float)testNode.x[1];
        ktestNode.mPoint.mPos[2] = (float)testNode.x[2];
        kt.nearest(kroot, &ktestNode, 0, kfound, bestDist);
        delete []nodes;
    }

    visited = 0;
    found = 0;
    nearest(root, &testNode, 0, 2, &found, &best_dist);

    printf(">> WP tree\nsearching for (%g, %g)\n"
            "found (%g, %g) dist %g\nseen %d nodes\n\n",
            testNode.x[0], testNode.x[1],
            found->x[0], found->x[1], sqrt(best_dist), visited);

    million =(struct kd_node_t*) calloc(N, sizeof(struct kd_node_t));
    srand(uint32_t(time(0)));
    for (i = 0; i < N; i++) rand_pt(million[i]);

    root = make_tree(million, N, 0, 3);
    rand_pt(testNode);

    visited = 0;
    found = 0;
    nearest(root, &testNode, 0, 3, &found, &best_dist);

    printf(">> Million tree\nsearching for (%g, %g, %g)\n"
            "found (%g, %g, %g) dist %g\nseen %d nodes\n",
            testNode.x[0], testNode.x[1], testNode.x[2],
            found->x[0], found->x[1], found->x[2],
            sqrt(best_dist), visited);

    /* search many random points in million tree to see average behavior.
       tree size vs avg nodes visited:
       10      ~  7
       100     ~ 16.5
       1000        ~ 25.5
       10000       ~ 32.8
       100000      ~ 38.3
       1000000     ~ 42.6
       10000000    ~ 46.7              */
    int sum = 0, test_runs = 100000;
    for (i = 0; i < test_runs; i++) {
        found = 0;
        visited = 0;
        rand_pt(testNode);
        nearest(root, &testNode, 0, 3, &found, &best_dist);
        sum += visited;
    }
    printf("\n>> Million tree\n"
            "visited %d nodes for %d random findings (%f per lookup)\n",
            sum, test_runs, sum/(double)test_runs);

    // free(million);

    return 0;
}

int main(int argc,const char **argv)
{
    testKdTree();

    uint32_t pointCount = 50000;
    float *points = new float[pointCount*3];
    std::random_device rand_device;
    uint64_t seed = 0; //rand_device();
    // Using a very simple PRBS generator, parameters selected according to
    // https://en.wikipedia.org/wiki/Linear_congruential_generator#Parameters_in_common_use
    std::linear_congruential_engine<uint64_t, 6364136223846793005, 1442695040888963407, UINT64_MAX> rand_engine(seed);
    std::uniform_int_distribution<size_t> uniform_generator(0,10000);
    for (uint32_t i=0; i<pointCount*3; i++)
    {
        points[i] = (float) uniform_generator(rand_engine);
    }


    

    kmeans::Kmeans *kpp = kmeans::Kmeans::create();
    uint32_t resultPointCount;
    printf("Running Kmeans against:%d input points.\n", pointCount);
    {
        ScopedTime st("Kmeans Time");
        const float *results = kpp->compute(points, pointCount, 1000, resultPointCount);
    }
    kpp->release();
}
