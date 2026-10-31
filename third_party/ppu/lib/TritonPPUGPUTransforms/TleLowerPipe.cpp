/*
 * Copyright (c) 2026 T-Head Semiconductor Co., Ltd. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files
 * (the "Software"), to deal in the Software without restriction,
 * including without limitation the rights to use, copy, modify, merge,
 * publish, distribute, sublicense, and/or sell copies of the Software,
 * and to permit persons to whom the Software is furnished to do so,
 * subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

// Lowering of TLE pipes (tle.pipe.*) to PPU hardware named barriers.
//
// PPU has no mbarrier: a barrier generation completes when a fixed number of
// threads have arrived (bar.arrive) or synchronized (bar.sync) on it, and the
// completion of an asynchronous copy is only observable by the thread that
// issued it (cp.async.wait_group). A pipe of capacity C is therefore mapped to
//
//   full[g][s] (s < C): the writer arrives after it published stage s, reader
//                       group g synchronizes before reading it;
//   empty[s]   (s < C): the readers arrive after they released stage s, the
//                       writer synchronizes before it refills it.
//
// A full barrier counts the writer and one reader group: readers reach their
// waits in unrelated order, so a barrier shared by two of them could deadlock.
// An empty barrier counts the writer and all readers, which only arrive on it.
// The readers arrive on every empty[s] once at start, so the first acquire of
// each stage passes. Barrier ids must be constants, so a barrier access with a
// runtime stage is an scf.index_switch with one case per stage. A commit of a
// stage written by cp.async first commits a copy group and waits for it: the
// hardware barrier only publishes the warp's completed shared-memory stores.
// For a single mixed pipe with a reader in the writer's group, structured
// lowering instead queues the commits in scalar registers. Its local wait
// completes and publishes the oldest slot while younger groups stay in flight.
// Reusing a pending slot, blocking on another group, and leaving the writer
// region drain pending publications. Other pipes keep the immediate wait.
//
// A pipe has one writer and at most one reader per warp group, and each
// endpoint steps through the stages in order; an endpoint that runs the
// sequence more often than its peer waits on a barrier nobody arrives at.
//
// When the writer and the readers are the same threads (a pipe used for
// software pipelining within one warp group) there is nobody to hand over to:
// acquire and wait become partition-wide barriers, and the wait also drains
// the copy groups the same threads committed.
//
// Close: a pipe with writer_close keeps one i32 tag per stage in shared
// memory, as the NVIDIA lowering does; close acquires the stage, stores the
// tag and publishes the stage, and reader_wait reads it after the barrier.
//
// Hardware barriers are scarce (16 per CTA, two reserved by the warp
// specialization lowering plus one per partition, and the TLE named barriers
// of the kernel). Pipes get them while they fit, the ones in the deepest loops
// first and in creation order among equals; the others use software barriers
// in shared memory: one i32 counter per stage for the writer (full) and one
// per stage and reader group (empty). After a group-wide barrier the group
// increments its counter with a scalar atomic add, which the lowering executes
// on one thread. A waiting group spins until the counter's parity matches the
// phase of the stage (a writer cannot run more than one generation ahead of
// its readers, so parity is enough, as with an mbarrier). The spin loop holds
// no group barrier: each warp leaves it after its own number of iterations.

#include "TritonPPUGPUTransforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SmallBitVector.h"
#ifdef __TLE__
#include "tle/dialect/include/IR/Dialect.h"
#include "tle/dialect/include/Transforms/TransformAttrs.h"
#endif

#include <map>

namespace mlir {
#define GEN_PASS_DEF_TRITONPPUTLELOWERPIPE
#include "TritonPPUGPUTransforms/Passes.h.inc"

namespace {
namespace tt = triton;
namespace ttg = triton::gpu;
namespace ttng = triton::nvidia_gpu;

#ifdef __TLE__
namespace tle = triton::tle;

// TLE named barrier ids at or above this value are virtual; the TLE allocator
// maps them to physical ids at the end of the TTGIR pipeline.
constexpr int64_t kFirstVirtualNamedBarrierId = 16;
constexpr int64_t kNumHardwareBarriers = 16;
// Ids the warp specialization lowering reserves: the default group's barrier,
// the switch loop, and one per partition.
constexpr int64_t kNumWarpSpecializeReservedBarriers = 2;

// The threads that execute an op: a ttg.warp_specialize worker partition, or
// the default warps (the default region and everything outside the op).
struct Partition {
  Region *region = nullptr;

  bool operator==(const Partition &other) const {
    return region == other.region;
  }
  bool operator!=(const Partition &other) const { return !(*this == other); }
};

static Partition getPartition(Operation *op) {
  for (Region *region = op->getParentRegion(); region;) {
    Operation *parent = region->getParentOp();
    if (!parent || isa<tt::FuncOp>(parent))
      break;
    if (isa<ttg::WarpSpecializePartitionsOp>(parent))
      return {region};
    region = parent->getParentRegion();
  }
  return {};
}

static int getThreadCount(Operation *op) {
  auto module = op->getParentOfType<ModuleOp>();
  return ttg::lookupNumWarps(op) *
         ttg::TritonGPUDialect::getThreadsPerWarp(module);
}

// A value used inside a worker partition is a region argument standing for a
// capture of the enclosing ttg.warp_specialize.
static Value canonicalizeCapture(Value value) {
  while (auto arg = dyn_cast<BlockArgument>(value)) {
    auto partitions = dyn_cast_or_null<ttg::WarpSpecializePartitionsOp>(
        arg.getOwner()->getParentOp());
    if (!partitions)
      break;
    auto ws = cast<ttg::WarpSpecializeOp>(partitions->getParentOp());
    OperandRange captures = ws.getExplicitCaptures();
    if (arg.getArgNumber() >= captures.size())
      break;
    value = captures[arg.getArgNumber()];
  }
  return value;
}

// The number of loops around `op` inside its warp group.
static int getLoopDepth(Operation *op) {
  int depth = 0;
  for (Operation *parent = op->getParentOp();
       parent && !isa<tt::FuncOp, ttg::WarpSpecializePartitionsOp>(parent);
       parent = parent->getParentOp())
    depth += isa<scf::ForOp, scf::WhileOp>(parent);
  return depth;
}

static Value getMemDescRoot(Value value) {
  value = canonicalizeCapture(value);
  while (Operation *def = value.getDefiningOp()) {
    if (!isa<ttg::MemDescIndexOp, ttg::MemDescSubsliceOp, ttg::MemDescTransOp,
             ttg::MemDescReinterpretOp, tle::MemDescAliasOp>(def))
      break;
    value = canonicalizeCapture(def->getOperand(0));
  }
  return value;
}

// Makes `value` available at `useOp`, capturing it into the enclosing worker
// partition when it is defined outside.
static Value captureForUse(Operation *useOp, Value value) {
  Partition partition = getPartition(useOp);
  if (!partition.region)
    return value;
  if (auto arg = dyn_cast<BlockArgument>(value)) {
    if (partition.region->isAncestor(arg.getOwner()->getParent()))
      return value;
  } else if (partition.region->isAncestor(
                 value.getDefiningOp()->getParentRegion())) {
    return value;
  }
  auto partitions =
      cast<ttg::WarpSpecializePartitionsOp>(partition.region->getParentOp());
  auto ws = cast<ttg::WarpSpecializeOp>(partitions->getParentOp());
  for (auto [index, capture] : llvm::enumerate(ws.getExplicitCaptures()))
    if (capture == value)
      return partition.region->getArgument(index);
  ws->insertOperands(ws.getNumOperands(), value);
  for (Region *region : ws.getPartitionRegions())
    region->addArgument(value.getType(), value.getLoc());
  return partition.region->getArgument(ws.getNumOperands() - 1);
}

static OperandRange getPipeFields(Operation *op) {
  if (auto pipe = dyn_cast<tle::PipeCreateOp>(op))
    return pipe.getFields();
  if (auto pipe = dyn_cast<tle::PipeWriterAcquireOp>(op))
    return pipe.getFields();
  if (auto pipe = dyn_cast<tle::PipeWriterCommitOp>(op))
    return pipe.getFields();
  if (auto pipe = dyn_cast<tle::PipeWriterCloseOp>(op))
    return pipe.getFields();
  if (auto pipe = dyn_cast<tle::PipeReaderWaitOp>(op))
    return pipe.getFields();
  return cast<tle::PipeReaderReleaseOp>(op).getFields();
}

static bool isPipeLifecycleOp(Operation *op) {
  return isa<tle::PipeCreateOp, tle::PipeWriterAcquireOp,
             tle::PipeWriterCommitOp, tle::PipeWriterCloseOp,
             tle::PipeReaderWaitOp, tle::PipeReaderReleaseOp>(op);
}

// The lifecycle ops of one pipe share capacity, scope, name, field names and
// field allocations. Fields are compared by their allocation, since a reader
// in a worker partition sees them through captures.
static std::string getPipeKey(Operation *op) {
  std::string key;
  llvm::raw_string_ostream os(key);
  os << op->getAttrOfType<IntegerAttr>("capacity").getInt() << "|";
  op->getAttr("scope").print(os);
  os << "|";
  if (Attribute name = op->getAttr("pipe_name"))
    name.print(os);
  os << "|";
  op->getAttr("field_names").print(os);
  os << "|";
  for (Value field : getPipeFields(op))
    os << getMemDescRoot(field).getAsOpaquePointer() << ",";
  return key;
}

static std::optional<int64_t> getConstantI32(Value value) {
  if (auto op = value.getDefiningOp<arith::ConstantIntOp>())
    return op.value();
  if (auto op = value.getDefiningOp<arith::ConstantOp>())
    if (auto attr = dyn_cast<IntegerAttr>(op.getValue()))
      return attr.getInt();
  return std::nullopt;
}

static Value i32Const(OpBuilder &builder, Location loc, int64_t value) {
  return arith::ConstantIntOp::create(builder, loc, value, 32);
}

static void createNamedBarrier(OpBuilder &builder, Location loc, int64_t id,
                               int threads, bool arrive) {
  Value idValue = i32Const(builder, loc, id);
  Value threadsValue = i32Const(builder, loc, threads);
  if (arrive)
    ttng::NamedBarrierArriveOp::create(builder, loc, idValue, threadsValue);
  else
    ttng::NamedBarrierWaitOp::create(builder, loc, idValue, threadsValue);
}

// Runs `emit` for the stage selected at runtime: directly for a constant
// stage, otherwise in an scf.index_switch with one case per stage.
static void forStage(OpBuilder &builder, Location loc, Value stage,
                     int64_t capacity,
                     llvm::function_ref<void(OpBuilder &, int64_t)> emit) {
  if (capacity == 1) {
    emit(builder, 0);
    return;
  }
  if (std::optional<int64_t> constant = getConstantI32(stage)) {
    emit(builder, ((*constant % capacity) + capacity) % capacity);
    return;
  }
  Value index =
      arith::IndexCastOp::create(builder, loc, builder.getIndexType(), stage);
  SmallVector<int64_t> cases;
  for (int64_t s = 0; s + 1 < capacity; ++s)
    cases.push_back(s);
  auto switchOp = scf::IndexSwitchOp::create(builder, loc, TypeRange{}, index,
                                             cases, cases.size());
  OpBuilder::InsertionGuard guard(builder);
  auto fill = [&](Region &region, int64_t s) {
    Block *block = builder.createBlock(&region);
    builder.setInsertionPointToStart(block);
    emit(builder, s);
    scf::YieldOp::create(builder, loc);
  };
  for (auto [i, s] : llvm::enumerate(cases))
    fill(switchOp.getCaseRegions()[i], s);
  fill(switchOp.getDefaultRegion(), capacity - 1);
}

struct Endpoint {
  Partition partition;
  int threads = 0;
  // A reader that runs in the writer's warp group: it is ordered after the
  // writer by the group's own barrier instead of a hand-over barrier.
  bool inWriterGroup = false;
  // Index of a reader among the readers in other groups.
  int crossGroupIndex = -1;
};

struct Pipe {
  tle::PipeCreateOp create;
  int64_t capacity = 0;
  bool oneShot = false;
  // Declared reader names; empty for the single default reader "".
  SmallVector<std::string> readerNames;
  // Lifecycle ops other than create, in program order.
  SmallVector<Operation *> ops;
  std::optional<Endpoint> writer;
  std::map<std::string, Endpoint> readers;
  bool hasClose = false;
  bool usesIsClosed = false;
  bool hasCpAsyncCommit = false;
  // Deepest loop nesting of the pipe's ops within their warp group.
  int loopDepth = 0;

  // Readers outside the writer's warp group take part in the hand-over
  // barriers; none means the pipe only orders accesses within one group.
  int crossGroupReaders = 0;
  // Hardware barriers: virtual named barrier ids full[g][s] =
  // fullBase + g * capacity + s for reader group g, and empty[s] =
  // emptyBase + s. Software barriers: counters full[s] = s and
  // empty[g][s] = (1 + g) * capacity + s of `softwareCounters`.
  bool softwareBarriers = false;
  int64_t fullBase = -1;
  int64_t emptyBase = -1;
  Value softwareCounters;
  // Threads of the writer, and of each reader in another group (by
  // crossGroupIndex). A full barrier counts the writer and one reader: the
  // readers wait at different points of their program, so a barrier that
  // counted all of them would make each reader wait for the others. An empty
  // barrier counts the writer and every reader, since the readers only arrive.
  int writerThreads = 0;
  SmallVector<int> readerThreads;
  int emptyThreads = 0;

  Value closeTags;
  ttg::MemDescType closeTagSlotType;
};

using PipeMap =
    llvm::MapVector<std::string, Pipe, std::map<std::string, unsigned>>;

static Attribute getCloseTagEncoding(MLIRContext *context, int64_t rank) {
  SmallVector<unsigned> order;
  for (int64_t dim = rank - 1; dim >= 0; --dim)
    order.push_back(static_cast<unsigned>(dim));
  return ttg::SwizzledSharedEncodingAttr::get(
      context, 1, 1, 1, order, ttg::CTAEncodingAttr::getDefault(context, rank));
}

// The tag tensor type depends on the warps of the partition that accesses it.
static RankedTensorType getCloseTagTensorType(Operation *op, OpBuilder &builder,
                                              ArrayRef<int64_t> shape) {
  MLIRContext *context = op->getContext();
  auto module = op->getParentOfType<ModuleOp>();
  Attribute encoding = ttg::getDefaultBlockedEncoding(
      context, shape, ttg::lookupNumWarps(op),
      ttg::TritonGPUDialect::getThreadsPerWarp(module),
      ttg::TritonGPUDialect::getNumCTAs(module));
  return RankedTensorType::get(shape, builder.getI32Type(), encoding);
}

static Value createCloseTagTensor(OpBuilder &builder, Location loc,
                                  RankedTensorType type, bool value) {
  Value scalar = i32Const(builder, loc, value ? 1 : 0);
  return tt::SplatOp::create(builder, loc, type, scalar);
}

static void createCloseTags(OpBuilder &builder, Pipe &pipe) {
  Operation *create = pipe.create.getOperation();
  Location loc = create->getLoc();
  MLIRContext *context = create->getContext();
  auto sharedMemorySpace = ttg::SharedMemorySpaceAttr::get(context);
  auto arrayType = ttg::MemDescType::get(
      {pipe.capacity, 1}, builder.getI32Type(), getCloseTagEncoding(context, 2),
      sharedMemorySpace, /*mutableMemory=*/true);
  pipe.closeTagSlotType = ttg::MemDescType::get(
      {1}, builder.getI32Type(), getCloseTagEncoding(context, 1),
      sharedMemorySpace, /*mutableMemory=*/true);
  if (llvm::isPowerOf2_64(pipe.capacity)) {
    Value init = createCloseTagTensor(
        builder, loc,
        getCloseTagTensorType(create, builder, {pipe.capacity, 1}), false);
    pipe.closeTags = ttg::LocalAllocOp::create(builder, loc, arrayType, init);
    return;
  }
  // A register tensor needs a power-of-two element count; clear the exact
  // number of stages one slot at a time.
  pipe.closeTags = ttg::LocalAllocOp::create(builder, loc, arrayType);
  Value init = createCloseTagTensor(
      builder, loc, getCloseTagTensorType(create, builder, {1}), false);
  for (int64_t stage = 0; stage < pipe.capacity; ++stage) {
    Value slot = ttg::MemDescIndexOp::create(
        builder, loc, pipe.closeTagSlotType, pipe.closeTags,
        i32Const(builder, loc, stage));
    ttg::LocalStoreOp::create(builder, loc, init, slot);
  }
}

static Value createCloseTagSlot(OpBuilder &builder, Location loc,
                                const Pipe &pipe, Value stage, Operation *use) {
  Value tags = captureForUse(use, pipe.closeTags);
  return ttg::MemDescIndexOp::create(builder, loc, pipe.closeTagSlotType, tags,
                                     stage);
}

static void storeCloseTag(OpBuilder &builder, Location loc, const Pipe &pipe,
                          Value stage, Operation *use) {
  Value slot = createCloseTagSlot(builder, loc, pipe, stage, use);
  Value tag = createCloseTagTensor(
      builder, loc, getCloseTagTensorType(use, builder, {1}), true);
  ttg::LocalStoreOp::create(builder, loc, tag, slot);
}

static Value loadCloseTag(OpBuilder &builder, Location loc, const Pipe &pipe,
                          Value stage, Operation *use) {
  Value slot = createCloseTagSlot(builder, loc, pipe, stage, use);
  Value tag = ttg::LocalLoadOp::create(
      builder, loc, getCloseTagTensorType(use, builder, {1}), slot);
  Value scalar = tt::UnsplatOp::create(builder, loc, builder.getI32Type(), tag);
  return arith::CmpIOp::create(builder, loc, arith::CmpIPredicate::ne, scalar,
                               i32Const(builder, loc, 0));
}

static LogicalResult recordEndpoint(std::optional<Endpoint> &endpoint,
                                    Operation *op, StringRef role) {
  Endpoint current{getPartition(op), getThreadCount(op)};
  if (!endpoint) {
    endpoint = current;
    return success();
  }
  if (endpoint->partition != current.partition)
    return op->emitOpError("pipe ")
           << role
           << " ops must run in one warp group; PPU pipes hand over "
              "between warp groups with hardware barriers";
  return success();
}

static FailureOr<std::string> getReaderName(const Pipe &pipe, Operation *op) {
  std::string name;
  if (auto attr = op->getAttrOfType<StringAttr>("reader_name"))
    name = attr.getValue().str();
  if (pipe.readerNames.empty()) {
    if (!name.empty())
      return op->emitOpError("names reader '")
                 << name << "' of a pipe without declared readers",
             failure();
    return name;
  }
  if (!llvm::is_contained(pipe.readerNames, name))
    return op->emitOpError("names reader '")
               << name << "' that the pipe does not declare",
           failure();
  return name;
}

// Collects the pipes of the module and checks what the lowering relies on.
static LogicalResult analyzePipes(ModuleOp module, PipeMap &pipes) {
  SmallVector<Operation *> ops;
  module.walk([&](Operation *op) {
    if (isPipeLifecycleOp(op))
      ops.push_back(op);
  });
  for (Operation *op : ops) {
    if (auto create = dyn_cast<tle::PipeCreateOp>(op)) {
      Pipe &pipe = pipes[getPipeKey(op)];
      if (pipe.create)
        return create.emitOpError("duplicates an earlier pipe.create");
      pipe.create = create;
      pipe.capacity = create.getCapacity();
      if (auto oneShot = create.getOneShotAttr())
        pipe.oneShot = oneShot.getValue();
      if (auto readers = create.getReadersAttr())
        for (Attribute reader : readers)
          pipe.readerNames.push_back(cast<StringAttr>(reader).getValue().str());
      if (create.getScope() != "cta")
        return create.emitOpError("PPU pipes support only scope='cta'");
    }
  }
  for (Operation *op : ops) {
    if (isa<tle::PipeCreateOp>(op))
      continue;
    auto it = pipes.find(getPipeKey(op));
    if (it == pipes.end() || !it->second.create)
      return op->emitOpError(
          "has no matching pipe.create in this function; PPU pipes must be "
          "created and used in one kernel after inlining");
    Pipe &pipe = it->second;
    pipe.ops.push_back(op);
    pipe.loopDepth = std::max(pipe.loopDepth, getLoopDepth(op));
    if (isa<tle::PipeWriterAcquireOp, tle::PipeWriterCommitOp,
            tle::PipeWriterCloseOp>(op)) {
      if (failed(recordEndpoint(pipe.writer, op, "writer")))
        return failure();
      pipe.hasClose |= isa<tle::PipeWriterCloseOp>(op);
      pipe.hasCpAsyncCommit |= op->hasAttr(tle::kTlePipeCommitCpAsyncAttr);
      continue;
    }
    FailureOr<std::string> name = getReaderName(pipe, op);
    if (failed(name))
      return failure();
    std::optional<Endpoint> endpoint;
    if (auto found = pipe.readers.find(*name); found != pipe.readers.end())
      endpoint = found->second;
    if (failed(recordEndpoint(endpoint, op, "reader")))
      return failure();
    pipe.readers[*name] = *endpoint;
    if (auto wait = dyn_cast<tle::PipeReaderWaitOp>(op))
      pipe.usesIsClosed |= !wait.getIsClosed().use_empty();
  }

  for (auto &[key, pipe] : pipes) {
    if (!pipe.writer || pipe.readers.empty())
      return pipe.create.emitOpError(
          "needs a writer and at least one reader in this kernel");
    pipe.writerThreads = pipe.writer->threads;
    pipe.emptyThreads = pipe.writer->threads;
    for (auto &[name, reader] : pipe.readers) {
      for (auto &[otherName, other] : pipe.readers)
        if (otherName != name && other.partition == reader.partition)
          return pipe.create.emitOpError("readers '")
                 << name << "' and '" << otherName
                 << "' run in the same warp group; one reader per group";
      reader.inWriterGroup = reader.partition == pipe.writer->partition;
      if (reader.inWriterGroup)
        continue;
      reader.crossGroupIndex = pipe.crossGroupReaders++;
      pipe.readerThreads.push_back(reader.threads);
      pipe.emptyThreads += reader.threads;
    }
  }
  return success();
}

// Hardware barriers left for pipes: all of them minus the ids the warp
// specialization lowering reserves, the physical ids in use and the virtual
// ids other TLE barriers already take.
static int64_t countFreeHardwareBarriers(ModuleOp module) {
  int64_t partitions = 0;
  module.walk([&](ttg::WarpSpecializeOp ws) {
    partitions = std::max<int64_t>(partitions, ws.getPartitionRegions().size());
  });
  llvm::SmallBitVector physical(kNumHardwareBarriers);
  llvm::DenseSet<int64_t> virtualIds;
  auto visit = [&](Value id) {
    std::optional<int64_t> constant = getConstantI32(id);
    if (!constant)
      return;
    if (*constant >= kFirstVirtualNamedBarrierId)
      virtualIds.insert(*constant);
    else if (*constant >= 0)
      physical.set(*constant);
  };
  module.walk([&](Operation *op) {
    if (auto wait = dyn_cast<ttng::NamedBarrierWaitOp>(op))
      visit(wait.getBar());
    else if (auto arrive = dyn_cast<ttng::NamedBarrierArriveOp>(op))
      visit(arrive.getBar());
  });
  for (int64_t id = 0; id < kNumWarpSpecializeReservedBarriers + partitions;
       ++id)
    physical.set(id);
  return kNumHardwareBarriers - physical.count() - virtualIds.size();
}

static int64_t findNextVirtualBarrierId(ModuleOp module) {
  int64_t next = kFirstVirtualNamedBarrierId;
  auto visit = [&](Value id) {
    if (std::optional<int64_t> constant = getConstantI32(id))
      next = std::max(next, *constant + 1);
  };
  module.walk([&](Operation *op) {
    if (auto wait = dyn_cast<ttng::NamedBarrierWaitOp>(op))
      visit(wait.getBar());
    else if (auto arrive = dyn_cast<ttng::NamedBarrierArriveOp>(op))
      visit(arrive.getBar());
  });
  return next;
}

// A mixed pipe can complete its copies when its local reader reaches a slot,
// leaving the younger groups in flight. Keep the queue in SSA registers, and
// drain it before blocking on another endpoint or reusing a pending slot. Limit
// this to one pipe and structured control flow: unrelated copy groups and
// arbitrary region control flow need a joint copy-group analysis.
static Region *getDeferredWriterRegion(Pipe &pipe, size_t numPipes) {
  if (numPipes != 1 || pipe.capacity < 2 || pipe.capacity > 4 || pipe.oneShot ||
      pipe.hasClose || !pipe.hasCpAsyncCommit || !pipe.crossGroupReaders ||
      !llvm::any_of(pipe.readers,
                    [](const auto &item) { return item.second.inWriterGroup; }))
    return nullptr;
  Region *region = nullptr;
  for (Operation *op : pipe.ops) {
    if (getPartition(op) != pipe.writer->partition)
      continue;
    if (isa<tle::PipeWriterCommitOp>(op) &&
        !op->hasAttr(tle::kTlePipeCommitCpAsyncAttr))
      return nullptr;
    Region *candidate = op->getParentRegion();
    while (isa<scf::ForOp, scf::IfOp>(candidate->getParentOp()))
      candidate = candidate->getParentOp()->getParentRegion();
    if (region && candidate != region)
      return nullptr;
    region = candidate;
  }
  if (!region || !region->hasOneBlock())
    return nullptr;
  bool supported = true;
  region->walk([&](Operation *op) {
    if ((op->getNumRegions() && !isa<scf::ForOp, scf::IfOp>(op)) ||
        isa<ttg::AsyncCommitGroupOp, ttg::AsyncWaitOp, tt::CallOp,
            tt::ElementwiseInlineAsmOp>(op) ||
        op->getName().getStringRef().contains("aiu") ||
        op->hasAttr("tt.load.async"))
      supported = false;
  });
  return supported ? region : nullptr;
}

class PipeLowering {
public:
  PipeLowering(Pipe &pipe) : pipe(pipe) {}

  void lowerCreate() {
    Operation *create = pipe.create.getOperation();
    OpBuilder builder(create);
    if (pipe.hasClose || (pipe.usesIsClosed && !pipe.oneShot))
      createCloseTags(builder, pipe);
    if (!pipe.crossGroupReaders)
      return;
    if (pipe.softwareBarriers) {
      createSoftwareCounters(builder);
      return;
    }
    if (pipe.oneShot)
      return;
    // Every stage starts empty: the readers arrive once on each empty barrier
    // before their first wait, which lets the writer's first acquire pass.
    for (auto &[name, reader] : pipe.readers) {
      if (reader.inWriterGroup)
        continue;
      if (reader.partition.region)
        builder.setInsertionPointToStart(&reader.partition.region->front());
      else
        builder.setInsertionPointAfter(create);
      for (int64_t stage = 0; stage < pipe.capacity; ++stage)
        createNamedBarrier(builder, create->getLoc(),
                           barrierId(/*full=*/false, stage, 0),
                           barrierThreads(/*full=*/false, 0),
                           /*arrive=*/true);
    }
  }

  void lower(Operation *op) {
    OpBuilder builder(op);
    Location loc = op->getLoc();
    currentOp = op;
    currentPhase = Value();
    if (auto acquire = dyn_cast<tle::PipeWriterAcquireOp>(op))
      currentPhase = acquire.getPhase();
    else if (auto close = dyn_cast<tle::PipeWriterCloseOp>(op))
      currentPhase = close.getPhase();
    else if (auto wait = dyn_cast<tle::PipeReaderWaitOp>(op))
      currentPhase = wait.getPhase();
    if (auto acquire = dyn_cast<tle::PipeWriterAcquireOp>(op)) {
      if (!pipe.oneShot)
        waitEmpty(builder, loc, acquire.getStage());
    } else if (auto commit = dyn_cast<tle::PipeWriterCommitOp>(op)) {
      commitCopies(builder, loc, op);
      publish(builder, loc, commit.getStage());
    } else if (auto close = dyn_cast<tle::PipeWriterCloseOp>(op)) {
      waitEmpty(builder, loc, close.getStage());
      storeCloseTag(builder, loc, pipe, close.getStage(), op);
      publish(builder, loc, close.getStage());
    } else if (auto wait = dyn_cast<tle::PipeReaderWaitOp>(op)) {
      waitFull(builder, loc, wait.getStage(), readerOf(op));
      if (!wait.getIsClosed().use_empty()) {
        Value isClosed =
            pipe.oneShot
                ? arith::ConstantIntOp::create(builder, loc, 0, 1).getResult()
                : loadCloseTag(builder, loc, pipe, wait.getStage(), op);
        wait.getIsClosed().replaceAllUsesWith(isClosed);
      }
    } else if (auto release = dyn_cast<tle::PipeReaderReleaseOp>(op)) {
      if (!pipe.oneShot && !readerOf(op).inWriterGroup)
        forStage(builder, loc, release.getStage(), pipe.capacity,
                 [&](OpBuilder &b, int64_t s) {
                   arrive(b, loc, /*full=*/false, s,
                          readerOf(op).crossGroupIndex);
                 });
    }
    op->erase();
  }

  void lowerDeferred(Region &region) {
    OpBuilder builder(&region.front(), region.front().begin());
    Location loc = pipe.create.getLoc();
    Pending pending{i32Const(builder, loc, 0), i32Const(builder, loc, 0)};
    pending = lowerDeferredBlock(region.front(), pending);
    Operation *end = region.front().getTerminator();
    builder.setInsertionPoint(end);
    drainIfPending(builder, end, pending);
  }

private:
  // Up to four stages, packed into two bits each, oldest in the low bits.
  struct Pending {
    Value count;
    Value stages;
  };

  Value queueStage(OpBuilder &b, Location loc, Pending pending, int index) {
    Value bits = pending.stages;
    if (index)
      bits = arith::ShRUIOp::create(b, loc, bits, i32Const(b, loc, 2 * index));
    return arith::AndIOp::create(b, loc, bits, i32Const(b, loc, 3));
  }

  Value queueContains(OpBuilder &b, Location loc, Pending pending,
                      Value stage) {
    Value found = arith::ConstantIntOp::create(b, loc, 0, 1);
    for (int i = 0; i < pipe.capacity; ++i) {
      Value occupied =
          arith::CmpIOp::create(b, loc, arith::CmpIPredicate::sgt,
                                pending.count, i32Const(b, loc, i));
      Value matches =
          arith::CmpIOp::create(b, loc, arith::CmpIPredicate::eq,
                                queueStage(b, loc, pending, i), stage);
      found = arith::OrIOp::create(
          b, loc, found, arith::AndIOp::create(b, loc, occupied, matches));
    }
    return found;
  }

  // Emit only constant wait counts, as required by cp.async.wait_group.
  void waitCopies(OpBuilder &b, Location loc, Value keep) {
    forStage(b, loc, keep, pipe.capacity, [&](OpBuilder &nested, int64_t n) {
      auto wait = ttg::AsyncWaitOp::create(
          nested, loc, ttg::AsyncTokenType::get(b.getContext()), ValueRange{},
          nested.getI32IntegerAttr(n));
      // The queue crosses structured joins through scalar registers rather
      // than async tokens. updateWaits cannot reconstruct its distance.
      wait->setAttr("ppugpu.pipe_wait", nested.getUnitAttr());
    });
  }

  // Complete the oldest `count - keep` groups and publish exactly those slots.
  Pending drain(OpBuilder &b, Operation *anchor, Pending pending, Value keep) {
    Location loc = anchor->getLoc();
    currentOp = anchor;
    waitCopies(b, loc, keep);
    Value completed = arith::SubIOp::create(b, loc, pending.count, keep);
    auto loop =
        scf::ForOp::create(b, loc, i32Const(b, loc, 0), completed,
                           i32Const(b, loc, 1), ValueRange{pending.stages});
    {
      OpBuilder::InsertionGuard guard(b);
      b.setInsertionPointToStart(loop.getBody());
      Value stages = loop.getRegionIterArgs()[0];
      Value stage = arith::AndIOp::create(b, loc, stages, i32Const(b, loc, 3));
      publish(b, loc, stage);
      Value rest = arith::ShRUIOp::create(b, loc, stages, i32Const(b, loc, 2));
      scf::YieldOp::create(b, loc, ValueRange{rest});
    }
    return {keep, loop.getResult(0)};
  }

  template <typename Emit>
  Pending conditionally(OpBuilder &b, Operation *anchor, Value condition,
                        Pending pending, Emit emit) {
    Location loc = anchor->getLoc();
    auto branch = scf::IfOp::create(
        b, loc, TypeRange{b.getI32Type(), b.getI32Type()}, condition, true);
    {
      OpBuilder::InsertionGuard guard(b);
      b.setInsertionPointToStart(&branch.getThenRegion().front());
      Pending next = emit(b, pending);
      scf::YieldOp::create(b, loc, ValueRange{next.count, next.stages});
      b.setInsertionPointToStart(&branch.getElseRegion().front());
      scf::YieldOp::create(b, loc, ValueRange{pending.count, pending.stages});
    }
    return {branch.getResult(0), branch.getResult(1)};
  }

  Pending drainIfPending(OpBuilder &b, Operation *anchor, Pending pending) {
    Value nonempty =
        arith::CmpIOp::create(b, anchor->getLoc(), arith::CmpIPredicate::ne,
                              pending.count, i32Const(b, anchor->getLoc(), 0));
    return conditionally(b, anchor, nonempty, pending,
                         [&](OpBuilder &nested, Pending p) {
                           return drain(nested, anchor, p,
                                        i32Const(nested, anchor->getLoc(), 0));
                         });
  }

  Pending consume(OpBuilder &b, Operation *anchor, Pending pending,
                  Value stage) {
    Location loc = anchor->getLoc();
    Value nonempty = arith::CmpIOp::create(b, loc, arith::CmpIPredicate::ne,
                                           pending.count, i32Const(b, loc, 0));
    Value oldest = arith::CmpIOp::create(b, loc, arith::CmpIPredicate::eq,
                                         queueStage(b, loc, pending, 0), stage);
    Value ready = arith::AndIOp::create(b, loc, nonempty, oldest);
    auto branch = scf::IfOp::create(
        b, loc, TypeRange{b.getI32Type(), b.getI32Type()}, ready, true);
    {
      OpBuilder::InsertionGuard guard(b);
      b.setInsertionPointToStart(&branch.getThenRegion().front());
      Value younger =
          arith::SubIOp::create(b, loc, pending.count, i32Const(b, loc, 1));
      waitCopies(b, loc, younger);
      currentOp = anchor;
      publish(b, loc, stage);
      Value rest =
          arith::ShRUIOp::create(b, loc, pending.stages, i32Const(b, loc, 2));
      scf::YieldOp::create(b, loc, ValueRange{younger, rest});
      b.setInsertionPointToStart(&branch.getElseRegion().front());
      Value found = queueContains(b, loc, pending, stage);
      Pending next = conditionally(
          b, anchor, found, pending, [&](OpBuilder &nested, Pending p) {
            return drain(nested, anchor, p, i32Const(nested, loc, 0));
          });
      scf::YieldOp::create(b, loc, ValueRange{next.count, next.stages});
    }
    return {branch.getResult(0), branch.getResult(1)};
  }

  Pending lowerDeferredBlock(Block &block, Pending pending) {
    SmallVector<Operation *> original;
    for (Operation &op : block.without_terminator())
      original.push_back(&op);
    for (Operation *op : original) {
      OpBuilder b(op);
      Location loc = op->getLoc();
      currentOp = op;
      if (auto loop = dyn_cast<scf::ForOp>(op)) {
        SmallVector<Value> init(loop.getInitArgs());
        init.append({pending.count, pending.stages});
        auto replacement =
            scf::ForOp::create(b, loc, loop.getLowerBound(),
                               loop.getUpperBound(), loop.getStep(), init);
        replacement->setAttrs(loop->getAttrs());
        // This explicit schedule owns the writer's copy groups. Later automatic
        // pipelining must not insert untracked groups between them.
        replacement->setAttr("tt.num_stages", b.getI32IntegerAttr(1));
        replacement.getBody()->erase();
        replacement.getRegion().getBlocks().splice(
            replacement.getRegion().getBlocks().begin(),
            loop.getRegion().getBlocks());
        replacement.getBody()->addArgument(b.getI32Type(), loc);
        replacement.getBody()->addArgument(b.getI32Type(), loc);
        auto args = replacement.getRegionIterArgs();
        Pending next = lowerDeferredBlock(*replacement.getBody(),
                                          {args[args.size() - 2], args.back()});
        replacement.getBody()->getTerminator()->insertOperands(
            loop.getNumResults(), ValueRange{next.count, next.stages});
        loop.replaceAllUsesWith(
            replacement.getResults().take_front(loop.getNumResults()));
        pending = {replacement.getResults()[loop.getNumResults()],
                   replacement.getResults().back()};
        loop.erase();
      } else if (auto branch = dyn_cast<scf::IfOp>(op)) {
        SmallVector<Type> types(branch.getResultTypes());
        types.append({b.getI32Type(), b.getI32Type()});
        auto replacement =
            scf::IfOp::create(b, loc, types, branch.getCondition(), true);
        replacement->setAttrs(branch->getAttrs());
        replacement.getThenRegion().takeBody(branch.getThenRegion());
        if (!branch.getElseRegion().empty())
          replacement.getElseRegion().takeBody(branch.getElseRegion());
        else {
          OpBuilder::InsertionGuard guard(b);
          b.setInsertionPointToStart(&replacement.getElseRegion().front());
          scf::YieldOp::create(b, loc);
        }
        for (Region *region :
             {&replacement.getThenRegion(), &replacement.getElseRegion()}) {
          Pending next = lowerDeferredBlock(region->front(), pending);
          region->front().getTerminator()->insertOperands(
              branch.getNumResults(), ValueRange{next.count, next.stages});
        }
        branch.replaceAllUsesWith(
            replacement.getResults().take_front(branch.getNumResults()));
        pending = {replacement.getResults()[branch.getNumResults()],
                   replacement.getResults().back()};
        branch.erase();
      } else if (auto acquire = dyn_cast<tle::PipeWriterAcquireOp>(op)) {
        Value reused = queueContains(b, loc, pending, acquire.getStage());
        pending = conditionally(
            b, op, reused, pending, [&](OpBuilder &nested, Pending p) {
              return drain(nested, op, p, i32Const(nested, loc, 0));
            });
        lower(op);
      } else if (auto commit = dyn_cast<tle::PipeWriterCommitOp>(op)) {
        ttg::AsyncCommitGroupOp::create(
            b, loc, ttg::AsyncTokenType::get(b.getContext()), ValueRange{});
        Value shift =
            arith::MulIOp::create(b, loc, pending.count, i32Const(b, loc, 2));
        Value stageBits =
            arith::ShLIOp::create(b, loc, commit.getStage(), shift);
        pending = {
            arith::AddIOp::create(b, loc, pending.count, i32Const(b, loc, 1)),
            arith::OrIOp::create(b, loc, pending.stages, stageBits)};
        op->erase();
      } else if (auto wait = dyn_cast<tle::PipeReaderWaitOp>(op)) {
        pending = consume(b, op, pending, wait.getStage());
        lower(op);
      } else if (isa<tle::PipeReaderReleaseOp>(op)) {
        lower(op);
      } else if (isa<ttng::NamedBarrierWaitOp, tt::AtomicRMWOp,
                     tt::AtomicCASOp>(op) ||
                 (isa<tt::LoadOp>(op) &&
                  cast<tt::LoadOp>(op).getIsVolatile())) {
        pending = drainIfPending(b, op, pending);
      }
    }
    return pending;
  }

  const Endpoint &readerOf(Operation *op) {
    std::string name;
    if (auto attr = op->getAttrOfType<StringAttr>("reader_name"))
      name = attr.getValue().str();
    return pipe.readers.at(name);
  }

  // Before the writer refills a stage: wait for its readers to release it. A
  // reader in the writer's group released it in program order.
  void waitEmpty(OpBuilder &builder, Location loc, Value stage) {
    if (pipe.oneShot)
      return;
    if (!pipe.crossGroupReaders) {
      ttg::LocalBarrierOp::create(builder, loc);
      return;
    }
    // The readers released the previous use of the stage; its generation has
    // the parity of this acquire's phase.
    forStage(builder, loc, stage, pipe.capacity, [&](OpBuilder &b, int64_t s) {
      wait(b, loc, /*full=*/false, s, /*readerIndex=*/0,
           /*invertPhase=*/false);
    });
  }

  // A stage written by cp.async is committed as one copy group, which the
  // writer drains before it publishes (the hardware barrier only covers the
  // warp's completed shared-memory stores). When only the writer's own group
  // reads the stage, it drains the group at its wait instead, so that the
  // copies of later stages stay in flight meanwhile.
  void commitCopies(OpBuilder &builder, Location loc, Operation *commit) {
    if (!commit->hasAttr(tle::kTlePipeCommitCpAsyncAttr))
      return;
    auto tokenType = ttg::AsyncTokenType::get(builder.getContext());
    Value group =
        ttg::AsyncCommitGroupOp::create(builder, loc, tokenType, ValueRange{});
    if (pipe.crossGroupReaders)
      ttg::AsyncWaitOp::create(builder, loc, tokenType, ValueRange{group},
                               builder.getI32IntegerAttr(0));
  }

  // After the writer filled a stage: let the readers in other groups go. Each
  // reader group has its own full barrier; software barriers share one counter.
  void publish(OpBuilder &builder, Location loc, Value stage) {
    if (!pipe.crossGroupReaders)
      return;
    forStage(builder, loc, stage, pipe.capacity, [&](OpBuilder &b, int64_t s) {
      if (pipe.softwareBarriers) {
        arrive(b, loc, /*full=*/true, s, /*readerIndex=*/0);
        return;
      }
      for (int reader = 0; reader < pipe.crossGroupReaders; ++reader)
        arrive(b, loc, /*full=*/true, s, reader);
    });
  }

  // Before a reader uses a stage: wait for the writer to publish it. In the
  // writer's group that is the group's barrier, after the group's own copies
  // when nobody drained them at the commit.
  void waitFull(OpBuilder &builder, Location loc, Value stage,
                const Endpoint &reader) {
    if (reader.inWriterGroup) {
      if (pipe.hasCpAsyncCommit && !pipe.crossGroupReaders)
        ttg::AsyncWaitOp::create(builder, loc,
                                 ttg::AsyncTokenType::get(builder.getContext()),
                                 ValueRange{}, builder.getI32IntegerAttr(0));
      ttg::LocalBarrierOp::create(builder, loc);
      return;
    }
    // The writer published this use of the stage: one generation beyond the
    // parity of the wait's phase.
    forStage(builder, loc, stage, pipe.capacity, [&](OpBuilder &b, int64_t s) {
      wait(b, loc, /*full=*/true, s, reader.crossGroupIndex,
           /*invertPhase=*/true);
    });
  }

  // The hardware barrier of a stage: full[reader group][stage] (the writer
  // arrives, that reader group synchronizes) or empty[stage] (every reader
  // group arrives, the writer synchronizes).
  int64_t barrierId(bool full, int64_t stage, int readerIndex) {
    if (full)
      return pipe.fullBase + readerIndex * pipe.capacity + stage;
    return pipe.emptyBase + stage;
  }

  // A full barrier counts the writer and one reader group, so a reader never
  // waits for the other readers, which reach their waits in a different order.
  // An empty barrier counts the writer and every reader group.
  int barrierThreads(bool full, int readerIndex) {
    if (full)
      return pipe.writerThreads + pipe.readerThreads[readerIndex];
    return pipe.emptyThreads;
  }

  // Arrival of the current op's group on full[reader group][stage] (the
  // writer) or on empty[stage] (the reader group `readerIndex`).
  void arrive(OpBuilder &b, Location loc, bool full, int64_t stage,
              int readerIndex) {
    if (!pipe.softwareBarriers) {
      createNamedBarrier(b, loc, barrierId(full, stage, readerIndex),
                         barrierThreads(full, readerIndex), /*arrive=*/true);
      return;
    }
    if (full)
      readerIndex = -1;
    // The group's accesses to the stage are complete before the counter
    // advances. A scalar atomic executes once per group, not per thread.
    ttg::LocalBarrierOp::create(b, loc);
    MLIRContext *ctx = b.getContext();
    tt::AtomicRMWOp::create(
        b, loc, b.getI32Type(), tt::RMWOpAttr::get(ctx, tt::RMWOp::ADD),
        counterPointer(b, loc, full, stage, readerIndex), i32Const(b, loc, 1),
        /*mask=*/Value(),
        tt::MemSemanticAttr::get(ctx, tt::MemSemantic::RELAXED),
        tt::MemSyncScopeAttr::get(ctx, tt::MemSyncScope::CTA));
  }

  // Wait until full[stage], or every empty[stage] counter, reaches the
  // generation whose parity is the current op's phase (inverted for full,
  // which the writer advances once before the readers' first wait).
  void wait(OpBuilder &b, Location loc, bool full, int64_t stage,
            int readerIndex, bool invertPhase) {
    if (!pipe.softwareBarriers) {
      createNamedBarrier(b, loc, barrierId(full, stage, readerIndex),
                         barrierThreads(full, readerIndex), /*arrive=*/false);
      return;
    }
    // Each warp leaves the spin loop after its own number of iterations, so the
    // loop must not contain a group barrier. This barrier settles, before the
    // loop, the shared-memory hazards of the counters (their zero fill, the
    // group's own counter updates) that the barrier analysis would otherwise
    // resolve with a barrier in front of the load inside the loop.
    ttg::LocalBarrierOp::create(b, loc);
    Value wanted = arith::ExtUIOp::create(b, loc, b.getI32Type(), currentPhase);
    if (invertPhase)
      wanted = arith::XOrIOp::create(b, loc, wanted, i32Const(b, loc, 1));
    int counters = full ? 1 : pipe.crossGroupReaders;
    for (int index = 0; index < counters; ++index) {
      Value counter = counterPointer(b, loc, full, stage, full ? -1 : index);
      auto whileOp = scf::WhileOp::create(b, loc, TypeRange{}, ValueRange{});
      OpBuilder::InsertionGuard guard(b);
      Block *before = b.createBlock(&whileOp.getBefore());
      b.setInsertionPointToStart(before);
      Value current = tt::LoadOp::create(
          b, loc, counter, tt::CacheModifier::NONE, tt::EvictionPolicy::NORMAL,
          /*isVolatile=*/true);
      Value parity =
          arith::AndIOp::create(b, loc, current, i32Const(b, loc, 1));
      Value pending = arith::CmpIOp::create(b, loc, arith::CmpIPredicate::ne,
                                            parity, wanted);
      scf::ConditionOp::create(b, loc, pending, ValueRange{});
      Block *after = b.createBlock(&whileOp.getAfter());
      b.setInsertionPointToStart(after);
      scf::YieldOp::create(b, loc);
    }
  }

  // The zeroed counters, allocated where the pipe is created.
  void createSoftwareCounters(OpBuilder &builder) {
    Operation *create = pipe.create.getOperation();
    Location loc = create->getLoc();
    MLIRContext *context = create->getContext();
    int64_t counters =
        (pipe.oneShot ? 1 : 1 + pipe.crossGroupReaders) * pipe.capacity;
    int64_t size = llvm::PowerOf2Ceil(counters);
    auto type = ttg::MemDescType::get(
        {size}, builder.getI32Type(), getCloseTagEncoding(context, 1),
        ttg::SharedMemorySpaceAttr::get(context), /*mutableMemory=*/true);
    Value zeros = createCloseTagTensor(
        builder, loc, getCloseTagTensorType(create, builder, {size}), false);
    pipe.softwareCounters =
        ttg::LocalAllocOp::create(builder, loc, type, zeros);
  }

  int64_t counterIndex(bool full, int64_t stage, int readerIndex) {
    return (full ? 0 : (1 + readerIndex) * pipe.capacity) + stage;
  }

  // The counter as a scalar shared-memory pointer.
  Value counterPointer(OpBuilder &b, Location loc, bool full, int64_t stage,
                       int readerIndex) {
    Value counters = captureForUse(currentOp, pipe.softwareCounters);
    return tle::LocalPointersOp::create(
        b, loc, tt::PointerType::get(b.getI32Type(), 3), counters,
        ValueRange{i32Const(b, loc, counterIndex(full, stage, readerIndex))});
  }

  Pipe &pipe;
  // The op being lowered and its phase operand (acquire, close, wait).
  Operation *currentOp = nullptr;
  Value currentPhase;
};
#endif // __TLE__

class TritonPPUTleLowerPipe
    : public impl::TritonPPUTleLowerPipeBase<TritonPPUTleLowerPipe> {
public:
  using impl::TritonPPUTleLowerPipeBase<
      TritonPPUTleLowerPipe>::TritonPPUTleLowerPipeBase;

  void runOnOperation() override {
#ifdef __TLE__
    ModuleOp module = getOperation();
    PipeMap pipes;
    if (failed(analyzePipes(module, pipes))) {
      signalPassFailure();
      return;
    }
    if (pipes.empty())
      return;

    int64_t nextId = findNextVirtualBarrierId(module);
    int64_t freeBarriers = countFreeHardwareBarriers(module);
    SmallVector<Pipe *> byPriority;
    for (auto &[key, pipe] : pipes)
      byPriority.push_back(&pipe);
    llvm::stable_sort(byPriority, [](const Pipe *a, const Pipe *b) {
      return a->loopDepth > b->loopDepth;
    });
    for (Pipe *pipePtr : byPriority) {
      Pipe &pipe = *pipePtr;
      if (!pipe.crossGroupReaders)
        continue;
      // One full barrier per stage and reader group, one empty barrier per
      // stage.
      int64_t fullBarriers = pipe.crossGroupReaders * pipe.capacity;
      int64_t needed = fullBarriers + (pipe.oneShot ? 0 : pipe.capacity);
      if (needed > freeBarriers) {
        pipe.softwareBarriers = true;
        continue;
      }
      freeBarriers -= needed;
      pipe.fullBase = nextId;
      nextId += fullBarriers;
      if (!pipe.oneShot) {
        pipe.emptyBase = nextId;
        nextId += pipe.capacity;
      }
    }

    for (auto &[key, pipe] : pipes) {
      PipeLowering lowering(pipe);
      lowering.lowerCreate();
      if (Region *writer = getDeferredWriterRegion(pipe, pipes.size())) {
        SmallVector<Operation *> remote;
        for (Operation *op : pipe.ops)
          if (!writer->isAncestor(op->getParentRegion()))
            remote.push_back(op);
        lowering.lowerDeferred(*writer);
        for (Operation *op : remote)
          lowering.lower(op);
      } else {
        for (Operation *op : pipe.ops)
          lowering.lower(op);
      }
      pipe.create.erase();
    }
#endif
  }
};

} // namespace
} // namespace mlir
