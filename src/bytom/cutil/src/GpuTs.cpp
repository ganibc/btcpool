#include <iostream>
#include <cstdio>
#include <map>
#include <mutex>
#include "GpuTs.h"
#include "BytomPoW.h"
#include "seed.h"
#include <cuda_runtime.h>
#include "cublas_v2.h"
#include <chrono>

using namespace std;

BytomMatList8* matList_int8;
BytomMatListGpu* matListGpu_int8;
uint8_t result[32] = {0};
map <vector<uint8_t>, BytomMatListGpu*> seedCache;
static const int cacheSize = 42; //"Answer to the Ultimate Question of Life, the Universe, and Everything"
mutex mtx;

uint8_t *SimdTs(uint8_t blockheader[32], uint8_t seed[32]){
    mtx.lock();
    auto s1 = chrono::high_resolution_clock::now();

    vector<uint8_t> seedVec(seed, seed + 32);

    if(seedCache.find(seedVec) != seedCache.end()) {
        // printf("\t---%s---\n", "Seed already exists in the cache.");
        matListGpu_int8 = seedCache[seedVec];
    } else {
        uint32_t exted[32];
        extend(exted, seed); // extends seed to exted
        auto s2 = chrono::high_resolution_clock::now();
        auto d1 = s2 - s1;
        std::cout << "d1 duration: " << chrono::duration_cast<chrono::microseconds>(d1).count() << " micros\n";

        Words32 extSeed;
        init_seed(extSeed, exted);
        auto s3 = chrono::high_resolution_clock::now();
        auto d2 = s3 - s2;
        std::cout << "d2 duration: " << chrono::duration_cast<chrono::microseconds>(d2).count() << " micros\n";

        matList_int8 = new BytomMatList8;
        matList_int8->init(extSeed);
        auto s4 = chrono::high_resolution_clock::now();
        auto d3 = s4 - s3;
        std::cout << "d3 duration: " << chrono::duration_cast<chrono::microseconds>(d3).count() << " micros\n";

        matListGpu_int8=new BytomMatListGpu;
        auto s5 = chrono::high_resolution_clock::now();
        auto d4 = s5 - s4;
        std::cout << "d4 duration: " << chrono::duration_cast<chrono::microseconds>(d4).count() << " micros\n";

        initMatVecGpu(matListGpu_int8, matList_int8);
        auto s6 = chrono::high_resolution_clock::now();
        auto d5 = s6 - s5;
        std::cout << "d5 duration: " << chrono::duration_cast<chrono::microseconds>(d5).count() << " micros\n";

        seedCache.insert(make_pair(seedVec, matListGpu_int8));

        delete matList_int8;
    }
    auto s7 = chrono::high_resolution_clock::now();
    // auto d6 = s7 - s6;
    // std::cout << "d6 duration: " << chrono::duration_cast<chrono::microseconds>(d6).count() << " micros\n";

    cublasHandle_t handle;
    cublasStatus_t stat = cublasCreate(&handle);
    if (stat != CUBLAS_STATUS_SUCCESS){
        std::cerr<<"Fail to Create CuBlas Handle."<<std::endl;
        exit(EXIT_FAILURE);
    }
    auto s8 = chrono::high_resolution_clock::now();
    auto d7 = s8 - s7;
    std::cout << "d7 duration: " << chrono::duration_cast<chrono::microseconds>(d7).count() << " micros\n";

    iter_mineBytom(blockheader, 32, result, handle);
    auto s9 = chrono::high_resolution_clock::now();
    auto d8 = s9 - s8;
    std::cout << "d8 duration: " << chrono::duration_cast<chrono::microseconds>(d8).count() << " micros\n";

    auto startSimdTs = chrono::high_resolution_clock::now();
    if(seedCache.size() > cacheSize) {
        for(map<vector<uint8_t>, BytomMatListGpu*>::iterator it=seedCache.begin(); it!=seedCache.end(); ++it){
            delete it->second; 
        }
        seedCache.clear();
        cudaDeviceReset();
    }
    auto s10 = chrono::high_resolution_clock::now();
    auto d9 = s10 - s9;
    std::cout << "d9 duration: " << chrono::duration_cast<chrono::microseconds>(d9).count() << " micros\n";

    mtx.unlock();
    
    return result;
}
