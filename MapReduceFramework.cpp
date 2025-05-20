#include <thread>
#include <mutex>
#include <vector>
#include <iostream>
#include "Barrier.h"

#include "MapReduceClient.h"
#include "MapReduceFramework.h"
#include <atomic>
#include <algorithm>

static constexpr int MAX_REASONABLE_THREAD_COUNT = 100000;
static constexpr uint64_t DONE_MASK = ((1ULL << 31) - 1) << 2;
static constexpr uint64_t TOTAL_MASK = ~((1ULL << 33) - 1);

struct JobContext;

typedef struct ThreadContext
{
    IntermediateVec *workingVec;
    JobContext *jobContext;
    int id;
} ThreadContext;

typedef struct JobContext
{
    std::vector<ThreadContext *> *threadsContext{};
    std::vector<std::thread *> *threads{};

    MapReduceClient const *client{};
    InputVec const *inputVec{};
    OutputVec *outputVec{};
    std::atomic<uint64_t> jobStatus{};
    bool isWaiting{};
    bool inUndefinedStage{};
    bool emptyInputVec{};

    std::mutex outputMtx;
    std::mutex undefStageMtx;
    std::mutex mapStageMtx;
    std::mutex reduceStageMtx;
    std::mutex inputVecMtx;
    Barrier *preBarrier{};
    Barrier *postBarrier{};

    std::atomic<int> inputPairsCounter{};

    std::vector<IntermediateVec *> *reduceQueue{};
    std::atomic<int> reduceQueueCounter{};

    std::atomic<uint64_t> newPairsCounter{};

} JobContext;

K2 *maxCurKeyOfAllThreads (JobContext *jobContext)
{

  K2 *curMax = nullptr;
  for (const auto &thread: *jobContext->threadsContext)
  {
    if (thread && thread->workingVec && !thread->workingVec->empty () &&
        (curMax == nullptr || *curMax < *thread->workingVec->back ().first))
    {
      curMax = thread->workingVec->back ().first;
    }
  }
  return curMax;
}

bool compareKeys (K2 *key1, K2 *key2)
{
  if (key1 == nullptr || key2 == nullptr)
  {
    return false;
  }
  if (*key1 < *key2 || *key2 < *key1)
  {
    return false;
  }
  else
  {
    return true;
  }
}

void moveToShuffleStage (ThreadContext *context)
{
  if (context->jobContext->emptyInputVec){
    uint64_t packed = (uint64_t(SHUFFLE_STAGE) & 0b11)|( (uint64_t)1 << 33)|
        ((uint64_t)1 << 2);
    context->jobContext->jobStatus.store (packed, std::memory_order_release);
  }
  else
  {
    uint64_t totalPairs = context->jobContext->newPairsCounter.load (std::memory_order_relaxed);
    uint64_t packed = (uint64_t (SHUFFLE_STAGE) & 0b11) | (totalPairs << 33);
    context->jobContext->jobStatus.store (packed, std::memory_order_release);
  }
}

void moveToReduceStage (ThreadContext *context)
{
  if (context->jobContext->emptyInputVec)
  {
    uint64_t packed =
        (uint64_t (REDUCE_STAGE) & 0b11) | ((uint64_t) 1 << 33) |
        ((uint64_t) 1 << 2);
    context->jobContext->jobStatus.store (packed, std::memory_order_release);
  }else
  {
    size_t totalSize = context->jobContext->reduceQueue->size ();
    uint64_t packedReduce =
        (uint64_t (REDUCE_STAGE) & 0b11)
        | (uint64_t (totalSize) << 33);
    context->jobContext->jobStatus.store (packedReduce, std::memory_order_release);
  }
}

void moveToMapStage (ThreadContext *context)
{
  if (context->jobContext->emptyInputVec)
  {
    uint64_t packed =
        (uint64_t (MAP_STAGE) & 0b11) | ((uint64_t) 1 << 33) |
        ((uint64_t) 1 << 2);
    context->jobContext->jobStatus.store (packed, std::memory_order_release);
  }
  else
  {
    size_t totalSize = ((context->jobContext->inputVec->size ()));
    uint64_t packedMap =
        (uint64_t (MAP_STAGE) & 0b11)
        | (uint64_t (totalSize) << 33);
    context->jobContext->jobStatus.store (packedMap,
                                          std::memory_order_release);
  }
}

void shuffleAll (ThreadContext *context)
{
  moveToShuffleStage (context);
  context->jobContext->reduceQueue = new std::vector<IntermediateVec *> ();

  bool remainingVectors = true;

  while (remainingVectors)
  {
    remainingVectors = false;
    K2 *maxKey = maxCurKeyOfAllThreads (context->jobContext);

    if (maxKey == nullptr)
    {
      break;
    }

    auto *newVec = new IntermediateVec ();

    for (auto &thread: *context->jobContext->threadsContext)
    {
      while (!thread->workingVec->empty () &&
             compareKeys (maxKey, thread->workingVec->back ().first))
      {

        newVec->push_back (thread->workingVec->back ());
        thread->workingVec->pop_back ();

        context->jobContext->jobStatus.fetch_add (
            1ULL << 2,
            std::memory_order_acq_rel

        );
      }
      if (!thread->workingVec->empty ())
      {
        remainingVectors = true;
      }
    }

    context->jobContext->reduceQueue->push_back (newVec);
    context->jobContext->reduceQueueCounter.fetch_add (1);
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
    moveToShuffleStage (context);
    shuffleAll (context);
    moveToReduceStage (context);
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
    moveToMapStage (context);
  }
  jobContext->undefStageMtx.unlock ();
}

void mapStageInRoutine (ThreadContext *context)
{
  JobContext *jobContext = context->jobContext;

  jobContext->mapStageMtx.lock ();
  int nextJob = jobContext->inputPairsCounter.fetch_add (-1);
  jobContext->mapStageMtx.unlock ();

  nextJob--;
  if (nextJob < 0)
  {
    sortAndShuffleVec (context);
    return;
  }
  else
  {
    jobContext->inputVecMtx.lock ();
    auto pair = jobContext->inputVec->at (nextJob);
    jobContext->inputVecMtx.unlock ();

    jobContext->jobStatus.fetch_add (1ULL << 2,
                                     std::memory_order_acq_rel);
    jobContext->client->map (pair.first, pair.second, context);
  }
}

void reduceStageInRoutine (ThreadContext *context)
{
  JobContext *jobContext = context->jobContext;

  jobContext->reduceStageMtx.lock ();
  int keyToReduce = jobContext->reduceQueueCounter.fetch_add (-1);
  jobContext->reduceStageMtx.unlock ();

  keyToReduce--;
  if (keyToReduce < 0)
  {
    return;
  }

  jobContext->reduceStageMtx.lock ();
  IntermediateVec *nextJob = jobContext->reduceQueue->at (keyToReduce);
  jobContext->reduceStageMtx.unlock ();

  jobContext->jobStatus.fetch_add (
      1ULL << 2,
      std::memory_order_acq_rel
  );

  jobContext->client->reduce (nextJob, context);
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
      if (jobContext->reduceQueueCounter <= 0)
      {
        return;
      }
      reduceStageInRoutine (context);
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
    exit (1);
  }

  auto *jobContext = new JobContext ();

  jobContext->client = &client;
  jobContext->inputVec = &inputVec;
  jobContext->outputVec = &outputVec;
  jobContext->inUndefinedStage = true;

  jobContext->inputPairsCounter.store ((int) jobContext->inputVec->size ());

  jobContext->preBarrier = new Barrier (multiThreadLevel);
  jobContext->postBarrier = new Barrier (multiThreadLevel);

  jobContext->jobStatus.store ((uint64_t) UNDEFINED_STAGE);
  jobContext->isWaiting = false;

  jobContext->threadsContext = new std::vector<ThreadContext *> ();
  jobContext->threads = new std::vector<std::thread *> ();
  jobContext->reduceQueueCounter.store (0);
  jobContext->newPairsCounter.store (0);
  jobContext->emptyInputVec = false;

  if (inputVec.empty ())
  {
    jobContext->emptyInputVec = true;
  }

  for (int i = 0; i < multiThreadLevel; i++)
  {
    auto *context = new ThreadContext{new IntermediateVec (), jobContext, i};
    auto *newThread = new (std::nothrow) std::thread (runRoutine, context);
    if (newThread == nullptr)
    {
      std::cout << "system error: thread's creation failed.\n";
      exit (1);
    }
    jobContext->threads->push_back (newThread);
    jobContext->threadsContext->push_back (context);

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

  for (auto &vec: *jobContext->reduceQueue)
  {
    delete vec;
  }

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
