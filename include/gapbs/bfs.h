// This code is based on commit 6ac1af of https://github.com/sbeamer/gapbs/blob/master/src/bfs.cc .
// The code is modified by the authors of Sppart so that it can be used with data structures of Sppart and it can compute the SSSP distance.

// Copyright (c) 2015, The Regents of the University of California (Regents)
// See LICENSE.txt for license details

#pragma once

#include "bitmap.h"
#include "pvector.h"
#include "sliding_queue.h"
#include "graph.hpp"

/*
GAP Benchmark Suite
Kernel: Breadth-First Search (BFS)
Author: Scott Beamer

Will return parent array for a BFS traversal from a source vertex

This BFS implementation makes use of the Direction-Optimizing approach [1].
It uses the alpha and beta parameters to determine whether to switch search
directions. For representing the frontier, it uses a SlidingQueue for the
top-down approach and a Bitmap for the bottom-up approach. To reduce
false-sharing for the top-down approach, thread-local QueueBuffer's are used.

To save time computing the number of edges exiting the frontier, this
implementation precomputes the degrees in bulk at the beginning by storing
them in parent array as negative numbers. Thus the encoding of parent is:
  parent[x] < 0 implies x is unvisited and parent[x] = -out_degree(x)
  parent[x] >= 0 implies x been visited

[1] Scott Beamer, Krste Asanović, and David Patterson. "Direction-Optimizing
    Breadth-First Search." International Conference on High Performance
    Computing, Networking, Storage and Analysis (SC), Salt Lake City, Utah,
    November 2012.
*/

namespace gapbs{
  typedef int32_t NodeID;

  template<class XADJ_INT, class FLOAT> // assuming FLOAT is float or double
  int64_t BUStep(const Sppart::Graph<XADJ_INT> &g, pvector<NodeID> &parent, Bitmap &front,
                Bitmap &next, FLOAT* const dists, FLOAT depth) {
    int64_t awake_count = 0;
    next.reset();
    // #pragma omp parallel for reduction(+ : awake_count) schedule(dynamic, 1024) // !!! This cause non-deterministic depth values
    #pragma omp parallel for reduction(+ : awake_count)
    for (NodeID u=0; u < g.nv; u++) {
      if (parent[u] < 0) {
        for (XADJ_INT k = g.xadj[u]; k < g.xadj[u+1]; ++k){
          const int v = g.adjncy[k];
          if (front.get_bit(v)) {
            parent[u] = v;
            dists[u] = depth;
            awake_count++;
            next.set_bit(u);
            break;
          }
        }
      }
    }
    return awake_count;
  }

  template<class XADJ_INT, class FLOAT> // assuming FLOAT is float or double
  int64_t TDStep(const  Sppart::Graph<XADJ_INT> &g, pvector<NodeID> &parent,
                SlidingQueue<NodeID> &queue, FLOAT* const dists, FLOAT depth) {
    int64_t scout_count = 0;
    #pragma omp parallel
    {
      QueueBuffer<NodeID> lqueue(queue);
      #pragma omp for reduction(+ : scout_count)
      for (auto q_iter = queue.begin(); q_iter < queue.end(); q_iter++) {
        NodeID u = *q_iter;
        for (XADJ_INT k = g.xadj[u]; k < g.xadj[u+1]; ++k){
          const int v = g.adjncy[k];
          NodeID curr_val = parent[v];
          if (curr_val < 0) {
            if (compare_and_swap(parent[v], curr_val, u)) {
              lqueue.push_back(v);
              dists[v] = depth;
              scout_count += -curr_val;
            }
          }
        }
      }
      lqueue.flush();
    }
    return scout_count;
  }

  void QueueToBitmap(const SlidingQueue<NodeID> &queue, Bitmap &bm) {
    #pragma omp parallel for
    for (auto q_iter = queue.begin(); q_iter < queue.end(); q_iter++) {
      NodeID u = *q_iter;
      bm.set_bit_atomic(u);
    }
  }

  template<class XADJ_INT>
  void BitmapToQueue(const Sppart::Graph<XADJ_INT> &g, const Bitmap &bm,
                    SlidingQueue<NodeID> &queue) {
    #pragma omp parallel
    {
      QueueBuffer<NodeID> lqueue(queue);
      #pragma omp for
      for (NodeID n=0; n < g.nv; n++)
        if (bm.get_bit(n))
          lqueue.push_back(n);
      lqueue.flush();
    }
    queue.slide_window();
  }

  template<class XADJ_INT>
  pvector<NodeID> InitParent(const Sppart::Graph<XADJ_INT> &g) {
    pvector<NodeID> parent(g.nv);
    #pragma omp parallel for
    for (NodeID n=0; n < g.nv; n++)
      parent[n] = g.degree(n) != 0 ? -g.degree(n) : -1;
    return parent;
  }

  template<class XADJ_INT, class FLOAT> // assuming FLOAT is float or double
  pvector<NodeID> DOBFS(const Sppart::Graph<XADJ_INT> &g, NodeID source, FLOAT* const dists, const bool force_top_down, int alpha = 15,
                        int beta = 18) {
    pvector<NodeID> parent = InitParent(g);
    parent[source] = source;
    SlidingQueue<NodeID> queue(g.nv);
    queue.push_back(source);
    queue.slide_window();
    Bitmap curr(g.nv);
    curr.reset();
    Bitmap front(g.nv);
    front.reset();
    int64_t edges_to_check = g.xadj[g.nv];
    int64_t scout_count = g.degree(source);

    FLOAT depth = 0.0;
    dists[source] = depth;

    while (!queue.empty()) {
      depth += 1.0;
      // if (scout_count > edges_to_check / alpha) {
      if ( !force_top_down && (scout_count > edges_to_check / alpha) ) {
        int64_t awake_count, old_awake_count;
        QueueToBitmap(queue, front);
        awake_count = queue.size();
        queue.slide_window();
        do {
          old_awake_count = awake_count;
          awake_count = BUStep(g, parent, front, curr, dists, depth);
          front.swap(curr);
        } while ((awake_count >= old_awake_count) ||
                (awake_count > g.nv / beta));
        BitmapToQueue(g, front, queue);
        scout_count = 1;
      } else {
        edges_to_check -= scout_count;
        scout_count = TDStep(g, parent, queue, dists, depth);
        queue.slide_window();
      }
    }
    #pragma omp parallel for
    for (NodeID n = 0; n < g.nv; n++)
      if (parent[n] < -1)
        parent[n] = -1;
    return parent;
  }
} // namespace
