// Copyright (c) 2020 Yasunori Futamura All Rights Reserved.

// This code is ported from metis-5.1.0/GKlib/gk_mkpqueue.h
#ifndef INCLUDE_SPPART_PRIORITY_QUEUE_HPP
#define INCLUDE_SPPART_PRIORITY_QUEUE_HPP

namespace Sppart {

    class PriorityQueue {
    private:
        struct KeyValue{
            int key;
            int val;
        };

        const int maxnodes;
        int nnodes;
        std::unique_ptr<KeyValue[]> heap;
        std::unique_ptr<int[]> locator;

    public:
        PriorityQueue(int maxnodes)
        : maxnodes(maxnodes), nnodes(0), heap(new KeyValue[maxnodes]), locator(new int[maxnodes])
        {
            #pragma omp parallel for
            for (int i = 0; i < maxnodes; ++i){
                locator[i] = -1;
            }
        }

        bool empty() const {
            return nnodes == 0;
        }

        void insert(int node, int key){
            // assert(locator[node] == -1);
            int i = nnodes++;
            while ( i > 0 ) {
                const int j = (i-1)>>1;
                if (key > heap[j].key) {
                    heap[i] = heap[j];
                    locator[heap[i].val] = i;
                    i = j;
                }else{
                    break;
                }
            }
            // assert(i >= 0);
            heap[i].key = key;
            heap[i].val = node;
            locator[node] = i;
            return;
        }

        void remove(int node){
            int i;
            // assert(locator[node] != -1);
            // assert(heap[locator[node]].val == node);
            i = locator[node];
            locator[node] = -1;

            if (--this->nnodes > 0 && heap[this->nnodes].val != node) {
                node   = heap[this->nnodes].val;
                const int newkey = heap[this->nnodes].key;
                const int oldkey = heap[i].key;
                if (newkey > oldkey) { // Filter-up
                    while (i > 0) {
                        const int j = (i-1)>>1;
                        if (newkey > heap[j].key) {
                            heap[i] = heap[j];
                            locator[heap[i].val] = i;
                            i = j;
                        }else{
                            break;
                        }
                    }
                } else { // Filter down
                    int j;
                    const int nnodes = this->nnodes;
                    while ((j=(i<<1)+1) < nnodes) {
                        if (heap[j].key > newkey) {
                            if (j+1 < nnodes && (heap[j+1].key > heap[j].key) ){
                                j++;
                            }                                
                            heap[i] = heap[j];
                            locator[heap[i].val] = i;
                            i = j;
                        } else if (j+1 < nnodes && (heap[j+1].key > newkey )) {
                            j++;
                            heap[i] = heap[j];
                            locator[heap[i].val] = i;
                            i = j;
                        }else{
                            break;
                        }
                    }                    
                }
                heap[i].key   = newkey;
                heap[i].val   = node;
                locator[node] = i;
            }
            return;
        }

        void update(int node, int newkey){
            int i;
            const int oldkey = heap[locator[node]].key;

            // assert(locator[node] != -1);
            // assert(heap[locator[node]].val == node);
            i = locator[node];

            if ( newkey > oldkey ) { // Filter-up
                while (i > 0) {
                    const int j = (i-1)>>1;\
                    if ( newkey > heap[j].key) {
                        heap[i] = heap[j];
                        locator[heap[i].val] = i;
                        i = j;
                    } else{
                        break;
                    }
                }
              } else { // Filter down
                int j;
                const int nnodes = this->nnodes;
                while ((j=(i<<1)+1) < nnodes) {
                    if (heap[j].key > newkey) {
                        if (j+1 < nnodes && heap[j+1].key > heap[j].key){
                          j++;
                        }
                        heap[i] = heap[j];
                        locator[heap[i].val] = i;
                        i = j;
                    } else if (j+1 < nnodes && heap[j+1].key > newkey) {
                        j++;
                        heap[i] = heap[j];
                        locator[heap[i].val] = i;
                        i = j;
                    } else{
                        break;
                    }
                }
              }
          heap[i].key   = newkey;
          heap[i].val   = node;
          locator[node] = i;
          return;
        }

        int get_top(){
            if (this->nnodes == 0){
                // TODO:: throw exception
                return -1;
            }
            this->nnodes--;

            const int vtx = heap[0].val;
            locator[vtx] = -1;

            int i;
            if ((i = this->nnodes) > 0) {
                const int key  = heap[i].key;
                const int node = heap[i].val;
                i = 0;
                int j;
                while ((j=2*i+1) < this->nnodes) {
                    if (heap[j].key > key) {
                        if (j+1 < this->nnodes && (heap[j+1].key > heap[j].key)){
                            j = j + 1;
                        }
                        heap[i] = heap[j];
                        locator[heap[i].val] = i;
                        i = j;
                    } else if (j+1 < this->nnodes && (heap[j+1].key > key)) {
                        j = j+1;
                        heap[i] = heap[j];
                        locator[heap[i].val] = i;
                        i = j;
                    } else {
                        break;
                    }
                }
                heap[i].key = key;
                heap[i].val = node;
                locator[node] = i;
            }
            return vtx;
        }

        int see_top_val(){
            return (this->nnodes == 0 ? -1 : heap[0].val);
        }

        int see_top_key(){
            // if (this->nnodes == 0){
            //     // Todo assert
            // }
            return heap[0].key;
        }

        bool contains(int node){
            return locator[node] != -1;
        }

        void reset(){
            #pragma omp parallel for
            for (int i = nnodes-1; i >= 0; --i){
                locator[heap[i].val] = -1;    
            }
            nnodes = 0;
        }

    }; // class
} // namespace

#endif //INCLUDE_SPPART_PRIORITY_QUEUE_HPP
