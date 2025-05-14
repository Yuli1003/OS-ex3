#include <thread>
#include <mutex>
#include <condition_variable>
#include <vector>
#include <iostream>
#include "Barrier.h"
#include "MapReduceClient.h"
#include "MapReduceFramework.h"
#include <atomic>
#include <queue>
#include "semaphore.h"
#include <algorithm>

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
    bool inUndefinedStage;

    std::mutex outputMtx;
    std::mutex undefStageMtx;
    std::mutex mapStageMtx;
    std::mutex shuffleMtx;
    std::mutex reduceStageMtx;

    Barrier *pre_barrier;
    Barrier *post_barrier;

    std::queue<IntermediateVec *> *reduceQueue;

    std::atomic<uint64_t> newPairsCounter;

} JobContext;

uint64_t doneJob (ThreadContext *context)
{
  return ((context->job_context->jobStatus.load () << 31) >> 33);
}

uint64_t totalJob (ThreadContext *context)
{
  return (context->job_context->jobStatus.load () >> 33);
}

K2 *maxCurKeyOfAllThreads (JobContext *jobContext)
{
  K2 *curMax = nullptr;
  for (auto &thread: *jobContext->threads_context)
  {
    if (thread->working_vec->empty ())
    {
      continue;
    }
    K2 *curLast = thread->working_vec->back ().first;
    if (curMax == nullptr || *curMax < *curLast)
    {
      curMax = curLast;
    }
  }
  return curMax;
}

void shuffleAll (ThreadContext *context)
{
  context->job_context->reduceQueue = new std::queue<IntermediateVec *> ();
  while (doneJob (context) < totalJob (context))
  {
    K2 *maxKey = maxCurKeyOfAllThreads (context->job_context);

    auto *newVec = new IntermediateVec ();
    for (auto &thread: *context->job_context->threads_context)
    {
      if (thread->working_vec->empty ())
      {
        //context->job_context->jobStatus += (1 << 2);
        continue;
      }
      while (!thread->working_vec->empty () &&
             !(*thread->working_vec->back ().first < *maxKey ||
              *maxKey < *thread->working_vec->back ().first))
      {
        newVec->push_back (thread->working_vec->back ());
        thread->working_vec->pop_back ();
        context->job_context->jobStatus += (1 << 2);
      }
    }

    context->job_context->reduceQueue->push (newVec);
  }
}

void sortAndShuffleVec (ThreadContext *context)
{
  context->job_context->newPairsCounter += context->working_vec->size ();
  std::sort (context->working_vec->begin (), context->working_vec->end (),
             [] (auto &a, auto &b)
             {
                 return *(a.first) < *(b.first);
             });

  context->job_context->pre_barrier->barrier ();

  if (context->id == 0)
  {
    if ((context->job_context->jobStatus.load () & 0b11) != SHUFFLE_STAGE)
    {
      context->job_context->jobStatus.store (SHUFFLE_STAGE);
      context->job_context->jobStatus +=
          ((context->job_context->newPairsCounter.load ()) << 33);

    }
    shuffleAll (context);

  }

  context->job_context->post_barrier->barrier ();
}

void runRoutine (ThreadContext *context)
{
  JobContext *jobContext = context->job_context;
  bool moreToMap = false;
  bool moreToReduce = false;
  bool sortedAndShuffled = false;
  while (true)
  {

    // undefined stage
    jobContext->undefStageMtx.lock ();
    if ((jobContext->jobStatus.load () & 0b11) == UNDEFINED_STAGE &&
        jobContext->inUndefinedStage)
    {
      jobContext->jobStatus += MAP_STAGE;
      jobContext->inUndefinedStage = false;
      jobContext->jobStatus += ((jobContext->input_vec->size ()) << 33);
    }
    jobContext->undefStageMtx.unlock ();

    // map stage

    if ((jobContext->jobStatus.load () & 0b11) == MAP_STAGE)
    {

      jobContext->mapStageMtx.lock ();
      uint64_t nextJob;
      if (doneJob (context) < totalJob (context))
      {
        moreToMap = true;
        nextJob = doneJob (context);
        jobContext->jobStatus += (1 << 2);
      }
      else
      {
        moreToMap = false;
      }
      jobContext->mapStageMtx.unlock ();

      if (moreToMap)
      {
        auto pair = jobContext->input_vec->at (nextJob);
        jobContext->client->map (pair.first, pair.second, context);
      }
      else
      {

        sortAndShuffleVec (context);

        sortedAndShuffled = true;
      }
    }

    // reduce stage
    context->job_context->reduceStageMtx.lock ();
    if ((context->job_context->jobStatus.load () & 0b11) != REDUCE_STAGE &&
        sortedAndShuffled)
    {
      context->job_context->jobStatus.store (REDUCE_STAGE);
      context->job_context->jobStatus +=
          ((context->job_context->newPairsCounter.load ()) << 33);

    }
    context->job_context->reduceStageMtx.unlock ();

    if ((jobContext->jobStatus.load () & 0b11) == REDUCE_STAGE)
    {
      jobContext->reduceStageMtx.lock ();
      IntermediateVec *nextJob;
      if (doneJob (context) < totalJob (context))
      {
        moreToReduce = true;
        nextJob = context->job_context->reduceQueue->front ();
        context->job_context->reduceQueue->pop ();
        jobContext->jobStatus += ((nextJob->size ()) << 2);
      }
      else
      {
        moreToReduce = false;
      }
      jobContext->reduceStageMtx.unlock ();

      if (moreToReduce)
      {
        jobContext->client->reduce (nextJob, context);
      }
      else
      {
        break;
      }
    }
  }
}

JobHandle startMapReduceJob (const MapReduceClient &client,
                             const InputVec &inputVec, OutputVec &outputVec,
                             int multiThreadLevel)
{
  auto *job_context = new JobContext ();

  job_context->client = &client;
  job_context->input_vec = &inputVec;
  job_context->output_vec = &outputVec;
  job_context->inUndefinedStage = true;

  job_context->pre_barrier = new Barrier (multiThreadLevel);
  job_context->post_barrier = new Barrier (multiThreadLevel);

  job_context->jobStatus.store ((uint64_t) UNDEFINED_STAGE);
  job_context->isWaiting = false;

  job_context->threads_context = new std::vector<ThreadContext *> ();
  job_context->threads = new std::vector<std::thread *> ();

  job_context->newPairsCounter.store (0);

  for (int i = 0; i < multiThreadLevel; i++)
  {
    auto *context = new ThreadContext{job_context, i, new IntermediateVec ()};
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

  state->stage = static_cast<stage_t>(jobContext->jobStatus.load () & 3);
  if (state->stage == UNDEFINED_STAGE)
  {
    state->percentage = 0;
  }
  else
  {
    uint64_t totalJob = (jobContext->jobStatus.load () >> 33);
    uint64_t doneJob = ((jobContext->jobStatus.load () << 31) >> 33);
    state->percentage = ((float) doneJob / (float) totalJob) * 100;
  }
}

void closeJobHandle (JobHandle job)
{
  if (job == nullptr)
  {
    std::cerr << "system error: Invalid JobHandle";
    return;
  }
  waitForJob (job);
  auto *jobContext = (JobContext *) job;

  for (auto &threadContext: *jobContext->threads_context)
  {
    delete threadContext->working_vec;
    delete threadContext;
  }
  for (auto &thread: *jobContext->threads)
  {
    delete thread;
  }

  delete jobContext->threads_context;
  delete jobContext->threads;
  delete jobContext->reduceQueue;
  delete jobContext->pre_barrier;
  delete jobContext->post_barrier;

  delete jobContext;
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
}
