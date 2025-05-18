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
#include <algorithm>

static constexpr int MAX_REASONABLE_THREAD_COUNT = 10000;
static constexpr uint64_t DONE_MASK = ((1ULL << 31) - 1) << 2;
static constexpr uint64_t TOTAL_MASK = ~((1ULL << 33) - 1);

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
    std::mutex reduceStageMtx;

    Barrier *pre_barrier;
    Barrier *post_barrier;

    std::queue<IntermediateVec *> *reduceQueue;

    std::atomic<uint64_t> newPairsCounter;

} JobContext;

uint64_t doneJob (ThreadContext *context)
{
//  return ((context->job_context->jobStatus.load () << 31) >> 33);
  uint64_t s = context->job_context->jobStatus.load ();
  return (s & DONE_MASK) >> 2;
}

uint64_t totalJob (ThreadContext *context)
{
//  return (context->job_context->jobStatus.load () >> 33);
  uint64_t s = context->job_context->jobStatus.load ();
  return (s & TOTAL_MASK) >> 33;
}

K2 *maxCurKeyOfAllThreads (JobContext *jobContext)
{
  K2 *curMax = nullptr;
  bool allEmpty = true;
  for (auto &thread: *jobContext->threads_context)
  {
    if (!thread->working_vec->empty ())
    {
        allEmpty = false;
        K2 *curLast = thread->working_vec->back ().first;
        if (curMax == nullptr || *curMax < *curLast)
        {
            curMax = curLast;
        }
    }

  }
  if (allEmpty){
      return nullptr;
  }
  return curMax;
}

void shuffleAll (ThreadContext *context)
{
  while (doneJob (context) < totalJob (context))
  {
    K2 *maxKey = maxCurKeyOfAllThreads (context->job_context);

    if (maxKey == nullptr){
        break;
    }

    auto *newVec = new IntermediateVec ();
    for (auto &thread: *context->job_context->threads_context)
    {
      if (thread->working_vec->empty ())
      {
        continue;
      }
      while (!thread->working_vec->empty () &&
             !(*thread->working_vec->back ().first < *maxKey ||
               *maxKey < *thread->working_vec->back ().first))
      {
        newVec->push_back (thread->working_vec->back ());
        thread->working_vec->pop_back ();
        context->job_context->jobStatus.fetch_add (
            1ULL << 2,
            std::memory_order_acq_rel
        );
      }
    }
    context->job_context->reduceQueue->push (newVec);
  }
}

void sortAndShuffleVec (ThreadContext *context)
{
  context->job_context->newPairsCounter.fetch_add (
      context->working_vec->size (),
      std::memory_order_relaxed);

  std::sort (context->working_vec->begin (), context->working_vec->end (),
             [] (IntermediatePair &a, IntermediatePair &b)
             {
                 return *(a.first) < *(b.first);
             });

  context->job_context->pre_barrier->barrier ();

  if (context->id == 0)
  {
    uint64_t packed = (uint64_t (SHUFFLE_STAGE) & 0b11)
                      | (context->job_context->newPairsCounter.load () << 33);
    context->job_context->jobStatus.store (packed, std::memory_order_release);

    shuffleAll (context);

    size_t totalSize = context->job_context->reduceQueue->size ();
    uint64_t packedReduce =
        (uint64_t (REDUCE_STAGE) & 0b11)
        | (uint64_t (totalSize) << 33);
    context->job_context->jobStatus.store (packedReduce,
                                           std::memory_order_release);
  }

  context->job_context->post_barrier->barrier ();
}

void undefinedStageInRoutine (ThreadContext *context)
{
  JobContext *jobContext = context->job_context;

  jobContext->undefStageMtx.lock ();
  if ((jobContext->jobStatus.load () & 0b11) == UNDEFINED_STAGE &&
      jobContext->inUndefinedStage)
  {
    jobContext->inUndefinedStage = false;
    size_t totalSize = ((jobContext->input_vec->size ()));
    uint64_t packedMap =
        (uint64_t (MAP_STAGE) & 0b11)
        | (uint64_t (totalSize) << 33);
    context->job_context->jobStatus.store (packedMap,
                                           std::memory_order_release);
  }
  jobContext->undefStageMtx.unlock ();
}

void mapStageInRoutine (ThreadContext *context)
{
  JobContext *jobContext = context->job_context;
  bool moreToMap = false;

  jobContext->mapStageMtx.lock ();
  uint64_t nextJob;
  if (doneJob (context) < totalJob (context))
  {
    moreToMap = true;
    nextJob = doneJob (context);
    jobContext->jobStatus.fetch_add (1ULL << 2,
                                     std::memory_order_acq_rel);
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
  }
}

bool reduceStageInRoutine (ThreadContext *context)
{
  JobContext *jobContext = context->job_context;
  bool moreToReduce = false;

  jobContext->reduceStageMtx.lock ();
  IntermediateVec *nextJob;
  if (doneJob (context) < totalJob (context))
  {
    moreToReduce = true;
    nextJob = context->job_context->reduceQueue->front ();
    context->job_context->reduceQueue->pop ();
    jobContext->jobStatus.fetch_add (
        1ULL << 2,
        std::memory_order_acq_rel
    );
  }
  else
  {
    moreToReduce = false;
  }
  jobContext->reduceStageMtx.unlock ();

  if (moreToReduce)
  {
    jobContext->client->reduce (nextJob, context);
    delete nextJob;
  }
  else
  {
    return false;
  }

  return true;
}

void runRoutine (ThreadContext *context)
{
  JobContext *jobContext = context->job_context;

  while (true)
  {
    undefinedStageInRoutine (context);

    if ((jobContext->jobStatus.load () & 0b11) == MAP_STAGE)
    {
      mapStageInRoutine (context);
    }

    if ((jobContext->jobStatus.load () & 0b11) == REDUCE_STAGE)
    {
      if (!reduceStageInRoutine (context)){
        break;
      }
    }
  }
}

JobHandle startMapReduceJob (const MapReduceClient &client,
                             const InputVec &inputVec, OutputVec &outputVec,
                             int multiThreadLevel)
{
  if (multiThreadLevel <= 0 || multiThreadLevel > MAX_REASONABLE_THREAD_COUNT)
  {
    std::cout << "system error: Invalid number of threads.\n";
    std::exit (1);
  }

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
  job_context->reduceQueue = new std::queue<IntermediateVec *> ();

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
  uint64_t s = jobContext->jobStatus.load (std::memory_order_acquire);

  state->stage = static_cast<stage_t>(s & 3);
  if (state->stage == UNDEFINED_STAGE)
  {
    state->percentage = 0.0f;
  }
  else
  {
    uint64_t done = (s & DONE_MASK) >> 2;
    uint64_t total = (s & TOTAL_MASK) >> 33;
    state->percentage = (total == 0 ? 0.0f
                                    : float (done) / float (total) * 100.0f);
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
