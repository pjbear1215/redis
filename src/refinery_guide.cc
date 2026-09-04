/*
 * Refinery (Branch A), D12 -- the C shim over the OBASE runtime.
 * See refinery_guide.h for the contract. Nothing here includes a Redis
 * header: the two type worlds meet only in refinery_obj.c.
 */

#include "refinery_guide.h"

#include "globalConfig.h"
#include "Guide.hpp"
#include "Sama.h"
#include "ObjectCollector.h"
#include "refinery/Snapshot.h"

#include <jemalloc/jemalloc.h>

#include <fcntl.h>
#include <unistd.h>

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <string>
#include <vector>
#include <new>
#include <type_traits>

/* Defined in crest/runtime/TagRuntime.cc. addToTAG is the ATC pin the OBASE
 * instrumentation pass emits next to every guided dereference; Redis has no
 * pass, so the shim calls it explicitly. */
extern "C" {
ActiveScopeGuard *createTAG();
void destroyTAG(ActiveScopeGuard *guard);
void addToTAG(ActiveScopeGuard *guard, void *varPtr);
}

int refinery_on = 0;
int refinery_reuse_blocks = 1;
int refinery_robj_reuse = 0;
int refinery_lru_copy = 1;
/* I1: how long refinery_fork_begin() may wait for a collector round before it
 * gives up and fails the fork. REFINERY_FORK_WAIT_MAX_S, seconds. */
static unsigned refinery_fork_wait_max_s = 120;

static Sama *g_refinery_sama = nullptr;
static ObjectCollector *g_refinery_oc = nullptr;
static std::atomic<uint8_t> g_refinery_mode{0};

/* One ActiveScopeGuard per thread, constructed in place so that entering a
 * scope costs no allocation. Redis executes commands on one thread, but the
 * bio threads reach freeStringObject(), so this stays thread-local. */
static thread_local std::aligned_storage<sizeof(ActiveScopeGuard),
                                         alignof(ActiveScopeGuard)>::type tl_scope_buf;
static thread_local int tl_scope_depth = 0;

/* Host-side counters (INFO refinery). The runtime's own counters live in
 * globalConfig.cc and are reported by refinery_heaps(). */
static std::atomic<uint64_t> g_values_adopted{0};
static std::atomic<uint64_t> g_values_released{0};
static std::atomic<uint64_t> g_adopt_bytes{0};
static std::atomic<uint64_t> g_release_bytes{0};
static std::atomic<uint64_t> g_forks{0};
/* I1 instrumentation: forks refused because a collector round outran the cap,
 * and the longest wait a fork that did proceed actually served out. */
static std::atomic<uint64_t> g_fork_wait_timeouts{0};
static std::atomic<uint64_t> g_fork_wait_ms_max{0};
/*
 * I2, measured over exactly the child's life rather than inferred from a flag.
 * The four placement counters are snapshotted at the instant
 * g_snapshot_child_alive goes up and compared when it comes down; anything that
 * moved in between is an object the collector relocated under a live snapshot.
 * Both hooks run on the main Redis thread and the snapshot is only armed on the
 * success path, so no lock is needed. It assumes one checkpoint child at a time
 * -- true for RDB and AOF rewrite, which are mutually exclusive; a concurrent
 * CHILD_TYPE_LDB would interleave the brackets, and Redis does not use one here.
 */
static uint64_t g_plc_at_fork[4] = {0, 0, 0, 0};
static bool g_plc_armed = false;
static std::atomic<uint64_t> g_placement_moved_under_child{0};
static std::atomic<uint64_t> g_placement_delta_under_child{0};

static void placementSnapshot(uint64_t out[4])
{
    out[0] = g_stat_promoted_hot.load(std::memory_order_relaxed);
    out[1] = g_stat_promoted_write.load(std::memory_order_relaxed);
    out[2] = g_stat_evicted_write_to_hot.load(std::memory_order_relaxed);
    out[3] = g_stat_demoted_cold.load(std::memory_order_relaxed);
}
/* Live bytes in managed blocks, counted at the jemalloc boundary (sallocx of
 * what refinery_alloc handed out, minus sallocx of what refinery_release
 * frees). refinery_disown() moves ownership between slots and must not move
 * this. Migration reallocates the same size class in another arena, so a
 * relocation is invisible here -- which is what makes it the right number for
 * used_memory. */
static std::atomic<int64_t> g_managed_bytes{0};

static unsigned envUInt(const char *name, unsigned dflt)
{
    const char *v = getenv(name);
    if (v == nullptr || *v == '\0')
        return dflt;
    return (unsigned)strtoul(v, nullptr, 10);
}

static long nowMs()
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000L + ts.tv_nsec / 1000000L;
}

/*
 * globalConfig.cc computes the SODA bitmap's covered span from /proc/self/maps
 * at static-initialisation time: global_addr_start is the [heap] base and the
 * bitmap spans [heap, stack). If no brk-backed heap exists yet the span is 0
 * and every guide registration is silently dropped. Force a brk allocation
 * before any dynamic initialiser runs (priority 101 puts this ahead of all
 * default-priority constructors in the same binary).
 */
__attribute__((constructor(101))) static void refineryForceBrkHeap()
{
    void *p = malloc(64);
    /* touch it so the allocation cannot be optimised away, then keep it:
     * freeing could return the brk and remove [heap] again. */
    if (p)
        memset(p, 0, 64);
}

extern "C" int refinery_boot(void)
{
    const char *e = getenv("REFINERY");
    refinery_on = (e && (*e == '1' || *e == 'y' || *e == 'Y')) ? 1 : 0;
    {
        /* Read before the REFINERY=0 return: B1 is a Redis-side knob and has
         * to be settable on the stock arm too, as its own control. */
        const char *l = getenv("REFINERY_LRU_COPY");
        refinery_lru_copy = (l && *l == '0') ? 0 : 1;
    }
    if (!refinery_on)
        return 0;
    refineryInitFromEnv();
    {
        const char *r = getenv("REFINERY_REDIS_REUSE");
        refinery_reuse_blocks = (r && *r == '0') ? 0 : 1;
        const char *o = getenv("REFINERY_REDIS_ROBJ_REUSE");
        refinery_robj_reuse = (o && *o == '1' && refinery_reuse_blocks) ? 1 : 0;
        refinery_fork_wait_max_s = envUInt("REFINERY_FORK_WAIT_MAX_S", 120);
    }
    return 1;
}

extern "C" void refinery_start(void)
{
    if (!refinery_on || g_refinery_sama != nullptr)
        return;

    /* COLD (4 KiB), HOT (2 MiB THP) and, with the write axis on, WRITE.
     * Same rule as ProcessCommand.h: the hotness-only ablation must reserve
     * exactly what stock OBASE reserves, or the arms differ in fork cost. */
    std::vector<MemType> types = {MemType::DRAM, MemType::DRAM_2MB_THP};
    if (g_refinery_write_axis)
        types.push_back(MemType::DRAM_2MB_THP_W);
    g_refinery_sama = new Sama(types);
    g_sama = g_refinery_sama;

    g_refinery_mode.store((uint8_t)envUInt("REFINERY_MODE", 2),
                          std::memory_order_release);
    unsigned window = envUInt("REFINERY_WINDOW_S", 20);
    unsigned cold_after = envUInt("REFINERY_COLD_AFTER_N", 4);
    g_refinery_oc = ObjectCollector::initializeIfNeeded(
        window, (uint8_t)cold_after, nullptr, g_refinery_sama, &g_refinery_mode);
}

extern "C" void refinery_set_mode(int mode)
{
    g_refinery_mode.store((uint8_t)mode, std::memory_order_release);
}

extern "C" int refinery_get_mode(void)
{
    return (int)g_refinery_mode.load(std::memory_order_acquire);
}

/* ---- the guide word --------------------------------------------------- */

extern "C" uint64_t refinery_guide_deref(void **slot)
{
    Guide<void> *g = reinterpret_cast<Guide<void> *>(slot);
    if (__builtin_expect(tl_snapshot_child, 0))
        return g->getCanonicalAddress();
    addToTAG(nullptr, slot);
    return g->touchAndGetAddr();
}

extern "C" uint64_t refinery_guide_deref_write(void **slot)
{
    Guide<void> *g = reinterpret_cast<Guide<void> *>(slot);
    if (__builtin_expect(tl_snapshot_child, 0))
        return g->getCanonicalAddress();
    addToTAG(nullptr, slot);
    return g->touchWriteAndGetAddr();
}

extern "C" uint64_t refinery_guide_deref_ro(void **slot)
{
    return reinterpret_cast<Guide<void> *>(slot)->getCanonicalAddress();
}

extern "C" void refinery_mark_write(void **slot)
{
    if (__builtin_expect(tl_snapshot_child, 0))
        return;
    reinterpret_cast<Guide<void> *>(slot)->setWriteBit();
}

/* ---- managed blocks ---------------------------------------------------- */

extern "C" void *refinery_alloc(size_t n)
{
    if (!refinery_on)
        return nullptr;
    void *p = jem_malloc(n);
    if (p)
        g_managed_bytes.fetch_add((int64_t)jem_sallocx(p, 0),
                                  std::memory_order_relaxed);
    return p;
}

extern "C" size_t refinery_managed_bytes(void)
{
    int64_t v = g_managed_bytes.load(std::memory_order_relaxed);
    return v > 0 ? (size_t)v : 0;
}

extern "C" void refinery_install(void **slot, void *p)
{
    *reinterpret_cast<uintptr_t *>(slot) = 0;
    Guide<void> *g = reinterpret_cast<Guide<void> *>(slot);
    *g = p; /* SODA Set1 on &slot, then touchAndGetAddr() */
}

extern "C" void refinery_release(void **slot)
{
    Guide<void> *g = reinterpret_cast<Guide<void> *>(slot);
    if (!tl_snapshot_child)
    {
        /* Pin, and veto a migration that may already hold MLOCK on this
         * guide: touchAndGetAddr() clears MLOCK, which makes the collector's
         * publishing CAS fail, so it frees its copy instead of ours. */
        addToTAG(nullptr, slot);
        g->touchAndGetAddr();
    }
    {
        void *p = (void *)(uintptr_t)g->getCanonicalAddress();
        if (p)
            g_managed_bytes.fetch_sub((int64_t)jem_sallocx(p, 0),
                                      std::memory_order_relaxed);
    }
    g->destroy(); /* SODA Set0 + Sama::free into the object's own arena */
    *reinterpret_cast<uintptr_t *>(slot) = 0;
}

extern "C" void refinery_disown(void **slot)
{
    Guide<void> *g = reinterpret_cast<Guide<void> *>(slot);
    soda_.Set0(((uint64_t)&g->ptr - (uint64_t)global_addr_start) / 8);
    *reinterpret_cast<uintptr_t *>(slot) = 0;
}

extern "C" size_t refinery_block_size(const void *p)
{
    return p ? jem_sallocx(const_cast<void *>(p), 0) : 0;
}

/* ---- scopes ------------------------------------------------------------ */

extern "C" void refinery_scope_enter(void)
{
    if (!refinery_on)
        return;
    if (tl_scope_depth++ == 0)
        new (&tl_scope_buf) ActiveScopeGuard();
}

extern "C" void refinery_scope_exit(void)
{
    if (!refinery_on)
        return;
    if (--tl_scope_depth == 0)
        reinterpret_cast<ActiveScopeGuard *>(&tl_scope_buf)->~ActiveScopeGuard();
    if (tl_scope_depth < 0)
        tl_scope_depth = 0;
}

extern "C" void refinery_scope_reset(void)
{
    if (!refinery_on)
        return;
    while (tl_scope_depth > 0)
        refinery_scope_exit();
}

extern "C" void refinery_exit(int code)
{
    /*
     * D12 step 7: a plain exit() again. 4386eca had to _exit() because
     * ObjectCollector's migrator loop was `while (true)` and ~ObjectCollector
     * joined it unconditionally, so a process that built a collector hung in a
     * static destructor. The runtime now carries a stop flag set by the
     * destructor before the join and checked in the migrator loop, in its
     * sleep, and in signalMigrationBegin()'s quiesce wait, so the destructors
     * can run. refinery_scope_reset() at the four shutdown sites still has to
     * run first, or the collector is left waiting for a scope exit() will
     * never close. Requires an OBASE runtime with that fix linked in.
     */
    fflush(nullptr);
    exit(code);
}

/* ---- smaps sampling (E13) ---------------------------------------------- */

extern "C" void refinery_smaps_sample(const char *label)
{
    const char *dir = getenv("REFINERY_SNAP_DIR");
    if (dir == nullptr || *dir == '\0')
        return;
    char path[512];
    snprintf(path, sizeof(path), "%s/smaps.txt", dir);
    /* O_APPEND + a single writev-shaped copy: this runs inside redisFork()
     * between the collector quiesce and fork(), so it must not allocate and
     * must not take a lock another thread could hold. */
    int out = open(path, O_WRONLY | O_CREAT | O_APPEND, 0644);
    if (out < 0)
        return;
    char hdr[192];
    int n = snprintf(hdr, sizeof(hdr),
                     "=== REFINERY_SAMPLE label=%s t_ms=%ld pid=%d ===\n",
                     label, nowMs(), (int)getpid());
    ssize_t w = write(out, hdr, n);
    (void)w;
    int in = open("/proc/self/smaps", O_RDONLY);
    if (in >= 0)
    {
        char buf[65536];
        ssize_t r;
        while ((r = read(in, buf, sizeof(buf))) > 0)
        {
            ssize_t off = 0;
            while (off < r)
            {
                ssize_t k = write(out, buf + off, (size_t)(r - off));
                if (k <= 0)
                    break;
                off += k;
            }
        }
        close(in);
    }
    close(out);
}

/* ---- fork -------------------------------------------------------------- */

extern "C" unsigned refinery_fork_wait_max(void)
{
    return refinery_fork_wait_max_s;
}

extern "C" int refinery_fork_begin(void)
{
    if (!refinery_on)
        return 0;
    /*
     * I1: publish the fork intent, then wait for a migration round that is
     * already running. The collector publishes round_active before it reads
     * this flag (globalConfig.h), so at most one of the two proceeds.
     * g_snapshot_child_alive is raised here rather than after fork() because
     * the shim has no hook between fork() and the parent's return; the effect
     * is a strictly conservative I2 (no round may start from here until the
     * child is reaped).
     */
    g_fork_in_flight.store(true, std::memory_order_seq_cst);
    refineryNoteSnapshot(); /* opens the fork window (A2) + arms the CIW credit */

    /*
     * D12 step 7. The wait used to give up after 60 s and fork anyway ("never
     * block the server forever; I1 is then not provable"). Step 6's E13 run
     * caught what that costs: at 15 M objects a collector round outran the
     * cap, the fork landed on a live round, and both collector arms recorded
     * oc_round_started_with_child = 1 *with* real migration during the child's
     * life -- I1 violated, and the two arms had to be discarded. A silent
     * proceed turns a correctness invariant into a timing accident, so the cap
     * now FAILS the fork: redisFork() returns -1, rdbSaveBackground() sets
     * rdb_last_bgsave_status:err, and serverCron retries on its own schedule.
     * The refusal is transient by construction -- the round the fork is
     * waiting out is ended by the collector itself.
     */
    long start = nowMs();
    long cap_ms = (long)refinery_fork_wait_max_s * 1000L;
    while (g_oc_round_active.load(std::memory_order_seq_cst))
    {
        if (nowMs() - start > cap_ms)
        {
            /* Retract the intent. Leaving it set would make the collector skip
             * every round from here on (it reads this flag at round start).
             * The fork window is a timestamp that lapses on its own
             * (refineryInForkWindow), so nothing else has to be undone. */
            g_fork_in_flight.store(false, std::memory_order_seq_cst);
            g_fork_wait_timeouts.fetch_add(1, std::memory_order_relaxed);
            return -1;
        }
        struct timespec ts = {0, 2000000};
        nanosleep(&ts, nullptr);
    }
    {
        uint64_t waited = (uint64_t)(nowMs() - start);
        uint64_t prev = g_fork_wait_ms_max.load(std::memory_order_relaxed);
        while (waited > prev &&
               !g_fork_wait_ms_max.compare_exchange_weak(
                   prev, waited, std::memory_order_relaxed))
            ;
    }
    g_forks.fetch_add(1, std::memory_order_relaxed);
    /*
     * Only now, with no round in flight and fork_in_flight published, does a
     * child count as alive. Raising it before the wait made the collector's
     * end-of-round I2 check fire for a round the fork was correctly waiting
     * out -- a false positive on a counter whose whole job is to be zero.
     */
    placementSnapshot(g_plc_at_fork);
    g_plc_armed = true;
    g_snapshot_child_alive.store(true, std::memory_order_seq_cst);
    if (g_refinery_parent_ro_while_child)
        g_parent_ro_active.store(true, std::memory_order_seq_cst);
    return 0;
}

extern "C" void refinery_fork_child(void)
{
    if (!refinery_on)
        return;
    /* I3: the child must not write a guide word. REFINERY_RO_CHILD=0 leaves
     * the CAS in place, which is the control arm. */
    tl_snapshot_child = g_refinery_ro_child;
}

extern "C" void refinery_fork_end(void)
{
    if (!refinery_on)
        return;
    if (g_plc_armed)
    {
        uint64_t now4[4];
        placementSnapshot(now4);
        uint64_t moved = 0;
        for (int i = 0; i < 4; i++)
            moved += now4[i] - g_plc_at_fork[i];
        if (moved != 0)
        {
            g_placement_moved_under_child.fetch_add(1, std::memory_order_relaxed);
            g_placement_delta_under_child.fetch_add(moved, std::memory_order_relaxed);
        }
        g_plc_armed = false;
    }
    g_parent_ro_active.store(false, std::memory_order_seq_cst);
    g_snapshot_child_alive.store(false, std::memory_order_seq_cst);
    g_fork_in_flight.store(false, std::memory_order_seq_cst);
}

/* ---- introspection ----------------------------------------------------- */

extern "C" void refinery_note_adopt(size_t bytes)
{
    g_values_adopted.fetch_add(1, std::memory_order_relaxed);
    g_adopt_bytes.fetch_add(bytes, std::memory_order_relaxed);
}

extern "C" void refinery_note_release(size_t bytes)
{
    g_values_released.fetch_add(1, std::memory_order_relaxed);
    g_release_bytes.fetch_add(bytes, std::memory_order_relaxed);
}

static size_t copyOut(char *buf, size_t buflen, const std::string &s)
{
    size_t n = s.size();
    if (n >= buflen)
        n = buflen ? buflen - 1 : 0;
    if (buflen)
    {
        memcpy(buf, s.data(), n);
        buf[n] = '\0';
    }
    return n;
}

extern "C" size_t refinery_info(char *buf, size_t buflen)
{
    char line[512];
    std::string out;

    snprintf(line, sizeof(line),
             "refinery_enabled:%d\r\n"
             "refinery_mode:%d\r\n"
             "refinery_write_axis:%d\r\n"
             "refinery_ro_child:%d\r\n"
             "refinery_obase_hints:%d\r\n"
             "refinery_parent_ro_while_child:%d\r\n"
             "refinery_write_epoch_k:%u\r\n"
             "refinery_fork_window_s:%u\r\n"
             "refinery_fork_wait_max_s:%u\r\n"
             "refinery_reuse_blocks:%d\r\n"
             "refinery_robj_reuse:%d\r\n"
             "refinery_lru_copy:%d\r\n",
             refinery_on, refinery_get_mode(),
             g_refinery_write_axis ? 1 : 0, g_refinery_ro_child ? 1 : 0,
             g_refinery_obase_hints ? 1 : 0,
             g_refinery_parent_ro_while_child ? 1 : 0,
             g_refinery_write_epoch_k, g_refinery_fork_window_s,
             refinery_fork_wait_max_s,
             refinery_reuse_blocks, refinery_robj_reuse, refinery_lru_copy);
    out += line;

    /* Per-arena byte counts: current_size is what Sama has handed out, and
     * [base, used_end) is the range a pagemap scan must be clamped to. */
    static const char *hname[] = {"new", "hot", "cold", "write", "node"};
    /*
     * Signed on purpose. Sama's extent_dalloc_hook_dram subtracts `size` from
     * current_size and returns whatever the default dalloc hook returns; with
     * jemalloc's default opt_retain=true that hook reports failure and
     * jemalloc RETAINS the extent, recycling it later without calling the
     * alloc hook again. One alloc can therefore be followed by several
     * dallocs, so current_size drifts negative under a free-heavy workload --
     * which is how INFO printed 18446744073687187456 (= -21.3 MiB) in the
     * step-3/step-4 runs. The counter belongs to the shared runtime, so the
     * fix reported here is to stop printing an unsigned underflow and to
     * publish the arena's bump frontier, which is exact, alongside it.
     */
    int64_t heap_bytes[5] = {0, 0, 0, 0, 0};
    uint64_t heap_span[5] = {0, 0, 0, 0, 0};
    if (g_refinery_sama)
    {
        for (const auto &r : g_refinery_sama->getArenaRanges())
        {
            int idx = 0;
            switch (r.type)
            {
            case MemType::DRAM_2MB_THP: idx = 1; break;
            case MemType::DRAM: idx = 2; break;
            case MemType::DRAM_2MB_THP_W: idx = 3; break;
            case MemType::DRAM_NODE: idx = 4; break;
            default: idx = 0; break;
            }
            heap_bytes[idx] += (int64_t)r.current_size;
            heap_span[idx] += (r.pre_alloc > r.base) ? (r.pre_alloc - r.base) : 0;
            snprintf(line, sizeof(line),
                     "refinery_arena_%s_base:0x%llx\r\n"
                     "refinery_arena_%s_used_end:0x%llx\r\n",
                     hname[idx], (unsigned long long)r.base,
                     hname[idx], (unsigned long long)r.pre_alloc);
            out += line;
        }
    }
    snprintf(line, sizeof(line),
             "refinery_heap_hot_bytes:%lld\r\n"
             "refinery_heap_cold_bytes:%lld\r\n"
             "refinery_heap_write_bytes:%lld\r\n"
             "refinery_heap_new_span:%llu\r\n"
             "refinery_heap_hot_span:%llu\r\n"
             "refinery_heap_cold_span:%llu\r\n"
             "refinery_heap_write_span:%llu\r\n"
             "refinery_managed_bytes:%llu\r\n",
             (long long)heap_bytes[1],
             (long long)heap_bytes[2],
             (long long)heap_bytes[3],
             (unsigned long long)heap_span[0],
             (unsigned long long)heap_span[1],
             (unsigned long long)heap_span[2],
             (unsigned long long)heap_span[3],
             (unsigned long long)refinery_managed_bytes());
    out += line;

    snprintf(line, sizeof(line),
             "refinery_values_adopted:%llu\r\n"
             "refinery_values_released:%llu\r\n"
             "refinery_adopt_bytes:%llu\r\n"
             "refinery_release_bytes:%llu\r\n"
             "refinery_forks:%llu\r\n"
             "refinery_fork_wait_timeouts:%llu\r\n"
             "refinery_fork_wait_ms_max:%llu\r\n",
             (unsigned long long)g_values_adopted.load(),
             (unsigned long long)g_values_released.load(),
             (unsigned long long)g_adopt_bytes.load(),
             (unsigned long long)g_release_bytes.load(),
             (unsigned long long)g_forks.load(),
             (unsigned long long)g_fork_wait_timeouts.load(),
             (unsigned long long)g_fork_wait_ms_max.load());
    out += line;

    snprintf(line, sizeof(line),
             "refinery_promoted_hot:%llu\r\n"
             "refinery_promoted_write:%llu\r\n"
             "refinery_evicted_write_to_hot:%llu\r\n"
             "refinery_demoted_cold:%llu\r\n"
             "refinery_write_epoch_clears:%llu\r\n"
             "refinery_demote_held_by_fork_window:%llu\r\n"
             "refinery_oc_rounds_run:%llu\r\n"
             "refinery_oc_rounds_skipped_fork:%llu\r\n"
             "refinery_oc_round_started_with_child:%llu\r\n"
             "refinery_placement_moved_under_child:%llu\r\n"
             "refinery_placement_delta_under_child:%llu\r\n"
             "refinery_parent_ro_cas_suppressed:%llu\r\n",
             (unsigned long long)g_stat_promoted_hot.load(),
             (unsigned long long)g_stat_promoted_write.load(),
             (unsigned long long)g_stat_evicted_write_to_hot.load(),
             (unsigned long long)g_stat_demoted_cold.load(),
             (unsigned long long)g_stat_write_epoch_clears.load(),
             (unsigned long long)g_stat_demote_held_by_fork_window.load(),
             (unsigned long long)g_stat_oc_rounds_run.load(),
             (unsigned long long)g_stat_oc_rounds_skipped_fork.load(),
             (unsigned long long)g_stat_oc_round_started_with_child.load(),
             (unsigned long long)g_placement_moved_under_child.load(),
             (unsigned long long)g_placement_delta_under_child.load(),
             (unsigned long long)g_stat_parent_ro_cas_suppressed.load());
    out += line;

    return copyOut(buf, buflen, out);
}

extern "C" size_t refinery_heaps(char *buf, size_t buflen)
{
    return copyOut(buf, buflen, refineryHeapsReport());
}

extern "C" size_t refinery_dump_regions(char *buf, size_t buflen)
{
    return copyOut(buf, buflen, refineryDumpRegions());
}
