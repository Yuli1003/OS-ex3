#include <thread>
#include <mutex>
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
    JobContext *jobContext;
    int id;
    IntermediateVec *workingVec;
} ThreadContext;

typedef struct JobContext
{
    std::vector<ThreadContext *> *threadsContext;
    std::vector<std::thread *> *threads;

    MapReduceClient const *client;
    InputVec const *inputVec;
    OutputVec *outputVec;
    std::atomic<uint64_t> jobStatus;
    bool isWaiting;
    bool inUndefinedStage;

    std::mutex outputMtx;
    std::mutex undefStageMtx;
    std::mutex mapStageMtx;
    std::mutex reduceStageMtx;

    Barrier *preBarrier;
    Barrier *postBarrier;

    std::queue<IntermediateVec *> *reduceQueue;

    std::atomic<uint64_t> newPairsCounter;

} JobContext;

uint64_t doneJob (ThreadContext *context)
{
  uint64_t s = context->jobContext->jobStatus.load ();
  return (s & DONE_MASK) >> 2;
}

uint64_t totalJob (ThreadContext *context)
{
  uint64_t s = context->jobContext->jobStatus.load ();
  return (s & TOTAL_MASK) >> 33;
}

K2 *maxCurKeyOfAllThreads (JobContext *jobContext)
{
  K2 *curMax = nullptr;
  for (auto &thread: *jobContext->threadsContext)
  {
    if (thread->workingVec->empty ())
    {
      continue;
    }
    K2 *curLast = thread->workingVec->back ().first;
    if (curMax == nullptr || *curMax < *curLast)
    {
      curMax = curLast;
    }
  }
  return curMax;
}

void shuffleAll (ThreadContext *context)
{
  while (doneJob (context) < totalJob (context))
  {
    K2 *maxKey = maxCurKeyOfAllThreads (context->jobContext);

    if (maxKey == nullptr){
        break;
    }

    auto *newVec = new IntermediateVec ();
    for (auto &thread: *context->jobContext->threadsContext)
    {
      if (thread->workingVec->empty ())
      {
        continue;
      }
      while (!thread->workingVec->empty () &&
             !(*thread->workingVec->back ().first < *maxKey ||
               *maxKey < *thread->workingVec->back ().first))
      {

        newVec->push_back (std::move(thread->workingVec->back ()));
        thread->workingVec->pop_back ();
        context->jobContext->jobStatus.fetch_add (
            1ULL << 2,
            std::memory_order_acq_rel
        );
      }
    }

    context->jobContext->reduceQueue->push (newVec);
  }
}

void sortAndShuffleVec (ThreadContext *context)
{
  context->jobContext->newPairsCounter.fetch_add (
      context->workingVec->size (),
      std::memory_order_relaxed);

  std::sort (context->workingVec->begin (), context->workingVec->end (),
             [] (IntermediatePair &a, IntermediatePair &b)
             {
                 return *(a.first) < *(b.first);
             });

  context->jobContext->preBarrier->barrier ();

  if (context->id == 0)
  {
    uint64_t packed = (uint64_t (SHUFFLE_STAGE) & 0b11)
                      | (context->jobContext->newPairsCounter.load () << 33);
    context->jobContext->jobStatus.store (packed, std::memory_order_release);

    shuffleAll (context);

    size_t totalSize = context->jobContext->reduceQueue->size ();
    uint64_t packedReduce =
        (uint64_t (REDUCE_STAGE) & 0b11)
        | (uint64_t (totalSize) << 33);
    context->jobContext->jobStatus.store (packedReduce,
                                          std::memory_order_release);
  }

  context->jobContext->postBarrier->barrier ();
}

void undefinedStageInRoutine (ThreadContext *context)
{
  JobContext *jobContext = context->jobContext;

  jobContext->undefStageMtx.lock ();
  if ((jobContext->jobStatus.load () & 0b11) == UNDEFINED_STAGE &&
      jobContext->inUndefinedStage)
  {
    jobContext->inUndefinedStage = false;
    size_t totalSize = ((jobContext->inputVec->size ()));
    uint64_t packedMap =
        (uint64_t (MAP_STAGE) & 0b11)
        | (uint64_t (totalSize) << 33);
    context->jobContext->jobStatus.store (packedMap,
                                          std::memory_order_release);
  }
  jobContext->undefStageMtx.unlock ();
}

void mapStageInRoutine (ThreadContext *context)
{
  JobContext *jobContext = context->jobContext;
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
    auto pair = jobContext->inputVec->at (nextJob);
    jobContext->client->map (pair.first, pair.second, context);
  }
  else
  {
    sortAndShuffleVec (context);
  }
}

bool reduceStageInRoutine (ThreadContext *context)
{
  JobContext *jobContext = context->jobContext;
  bool moreToReduce = false;

  jobContext->reduceStageMtx.lock ();
  IntermediateVec *nextJob = nullptr;
  if (doneJob (context) < totalJob (context)
      && !jobContext->reduceQueue->empty ())
  {
    moreToReduce = true;
    nextJob = context->jobContext->reduceQueue->front ();
    context->jobContext->reduceQueue->pop ();
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
  JobContext *jobContext = context->jobContext;

  while (true)
  {
    undefinedStageInRoutine (context);

    if ((jobContext->jobStatus.load () & 0b11) == MAP_STAGE)
    {
      mapStageInRoutine (context);
    }

    if ((jobContext->jobStatus.load () & 0b11) == REDUCE_STAGE)
    {
      if (!reduceStageInRoutine (context))
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
  if (multiThreadLevel <= 0 || multiThreadLevel > MAX_REASONABLE_THREAD_COUNT)
  {
    std::cout << "system error: Invalid number of threads.\n";
    std::exit (1);
  }

  auto *jobContext = new JobContext ();

  jobContext->client = &client;
  jobContext->inputVec = &inputVec;
  jobContext->outputVec = &outputVec;
  jobContext->inUndefinedStage = true;

  jobContext->preBarrier = new Barrier (multiThreadLevel);
  jobContext->postBarrier = new Barrier (multiThreadLevel);

  jobContext->jobStatus.store ((uint64_t) UNDEFINED_STAGE);
  jobContext->isWaiting = false;

  jobContext->threadsContext = new std::vector<ThreadContext *> ();
  jobContext->threads = new std::vector<std::thread *> ();
  jobContext->reduceQueue = new std::queue<IntermediateVec *> ();

  jobContext->newPairsCounter.store (0);

  if (inputVec.empty ())
  {
    uint64_t packed = (uint64_t (REDUCE_STAGE) & 0b11)
                      | (uint64_t (1) << 33) | (uint64_t (1) << 2);
    jobContext->jobStatus.store (packed, std::memory_order_release);
  }
  else
  {
    for (int i = 0; i < multiThreadLevel; i++)
    {
      auto *context = new ThreadContext{jobContext, i,
                                        new IntermediateVec ()};
      auto *newThread = new (std::nothrow) std::thread (runRoutine, context);
      if (newThread == nullptr)
      {
        std::cout << "system error: thread's creation failed.\n";
        exit (1);
      }
      jobContext->threads->push_back (newThread);
      jobContext->threadsContext->push_back (context);
    }
  }

  return (void *) jobContext;
}

void waitForJob (JobHandle job)
{
  if (job == nullptr)
  {
    std::cout << "system error: Invalid JobHandle.\n";
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
    std::cout << "system error: Invalid JobHandle or JobState.\n";
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
    std::cout << "system error: Invalid JobHandle.\n";
    return;
  }
  waitForJob (job);
  auto *jobContext = (JobContext *) job;

  for (auto &threadContext: *jobContext->threadsContext)
  {
    delete threadContext->workingVec;
    delete threadContext;
  }
  for (auto &thread: *jobContext->threads)
  {
    delete thread;
  }

  delete jobContext->threadsContext;
  delete jobContext->threads;
  delete jobContext->reduceQueue;
  delete jobContext->preBarrier;
  delete jobContext->postBarrier;

  delete jobContext;
}

void emit2 (K2 *key, V2 *value, void *context)
{
  if (key == nullptr || value == nullptr || context == nullptr)
  {
    std::cout << "system error: Invalid args for emit2.\n";
    return;
  }

  auto threadContext = (ThreadContext *) context;
  threadContext->workingVec->emplace_back (key, value);
}

void emit3 (K3 *key, V3 *value, void *context)
{
  if (key == nullptr || value == nullptr || context == nullptr)
  {
    std::cout << "system error: Invalid args for emit3.\n";
    return;
  }
  auto threadContext = (ThreadContext *) context;
  threadContext->jobContext->outputMtx.lock ();
  threadContext->jobContext->outputVec->emplace_back (key, value);
  threadContext->jobContext->outputMtx.unlock ();
}
