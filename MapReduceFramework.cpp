#include <thread>        // std::thread
#include <mutex>         // std::mutex, std::lock_guard
#include <condition_variable> // std::condition_variable
#include <chrono>        // clocks and durations
#include <vector>
#include <iostream>
#include "Barrier.h"
#include "MapReduceClient.h"
#include "MapReduceFramework.h"
#include <atomic>
#include <queue>
#include "semaphore.h"

struct JobContext;

typedef struct ThreadContext
{
    JobContext *job_context;
    int id;
    IntermediateVec *working_vec;
} ThreadContext;

typedef struct JobContext
{
    std::vector<ThreadContext *> *threads_context;
    std::vector<std::thread *> *threads;

    MapReduceClient const *client;
    InputVec const *input_vec;
    OutputVec *output_vec;
    std::atomic<uint64_t> jobStatus;
    bool isWaiting;

    std::mutex inputMtx;
    std::mutex outputMtx;

    std::mutex undefStageMtx;
    std::mutex mapStageMtx;
    std::mutex reduceStageMtx;

    Barrier *pre_barrier;
    Barrier *post_barrier;

    std::atomic<size_t> nextMapIdx{0};
    std::mutex shuffleMtx;
    std::condition_variable shuffleCv;
    std::queue<IntermediateVec> shuffleQueue;

} JobContext;

void runRoutine (ThreadContext *context)
{
  // undefined stage

  // map stage

  // reduce stage
}

JobHandle startMapReduceJob (const MapReduceClient &client,
                             const InputVec &inputVec, OutputVec &outputVec,
                             int multiThreadLevel)
{
  // job state

  auto *job_context = new JobContext ();

  job_context->client = &client;
  job_context->input_vec = &inputVec;
  job_context->output_vec = &outputVec;

  job_context->pre_barrier = new Barrier (multiThreadLevel);
  job_context->post_barrier = new Barrier (multiThreadLevel);

  job_context->jobStatus = (uint64_t) UNDEFINED_STAGE;
  job_context->isWaiting = false;

  job_context->threads_context = new std::vector<ThreadContext *> ();
  job_context->threads = new std::vector<std::thread *> ();

  for (int i = 0; i < multiThreadLevel; i++)
  {
    auto *context = new ThreadContext{job_context, i, nullptr};
    job_context->threads->push_back (new std::thread (runRoutine, context));
    job_context->threads_context->push_back (context);
  }

  return (void *) job_context;
}

void waitForJob (JobHandle job)
{
  if (job == nullptr)
  {
    std::cerr << "system error: Invalid JobHandle";
    return;
  }
  auto jobContext = (JobContext *) job;
  if (!jobContext->isWaiting)
  {
    jobContext->isWaiting = true;
    for (auto &thread: *jobContext->threads)
    {
      thread->join ();
    }
  }
}

void getJobState (JobHandle job, JobState *state)
{
  if (job == nullptr || state == nullptr)
  {
    std::cerr << "system error: Invalid JobHandle or JobState";
    return;
  }

  auto jobContext = (JobContext *) job;
  uint64_t curStatus = jobContext->jobStatus;

  state->stage = static_cast<stage_t>(curStatus & 3);
  if (state->stage == UNDEFINED_STAGE)
  {
    state->percentage = 0;
  }
  else
  {
    uint64_t totalJob = (curStatus >> 33);
    uint64_t doneJob = ((curStatus << 31) >> 33);
    state->percentage = ((float) doneJob / (float) totalJob) * 100;
  }
}

void closeJobHandle (JobHandle job)
{

}

void emit2 (K2 *key, V2 *value, void *context)
{
  if (key == nullptr || value == nullptr || context == nullptr)
  {
    std::cerr << "system error: Invalid args for emit2";
    return;
  }
  auto threadContext = (ThreadContext *) context;
  threadContext->working_vec->emplace_back (key, value);
  threadContext->job_context->jobStatus += (1 << 2);
}

void emit3 (K3 *key, V3 *value, void *context)
{
  if (key == nullptr || value == nullptr || context == nullptr)
  {
    std::cerr << "system error: Invalid args for emit3";
    return;
  }
  auto threadContext = (ThreadContext *) context;
  threadContext->job_context->outputMtx.lock ();
  threadContext->job_context->output_vec->emplace_back (key, value);
  threadContext->job_context->outputMtx.unlock ();
  threadContext->job_context->jobStatus += (1 << 2);
}
