#include <thread>        // std::thread
#include <mutex>         // std::mutex, std::lock_guard
#include <condition_variable> // std::condition_variable
#include <chrono>        // clocks and durations
#include <vector>
#include <iostream>
#include "Barrier.h"
#include <atomic>


class Framework{
 private:
  MapReduceClient client;
  JobState state;
  OutputVec* output;


 public:
  JobHandle startMapReduceJob(const MapReduceClient& client,
                              const InputVec& inputVec, OutputVec& outputVec,
                              int multiThreadLevel){
    this->client = client;
    this->output = outputVec;


  }

};
