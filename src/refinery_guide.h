#ifndef REFINERY_GUIDE_H
#define REFINERY_GUIDE_H

/*
 * Refinery (Branch A), D12 -- C-callable shim over the OBASE runtime.
 *
 * The OBASE runtime (crest/runtime) is C++ with templates and spdlog; Redis is
 * C99. This header is the entire interface between them. Everything below is
 * a thin wrapper over one of:
 *
 *   Guide<void>            crest/runtime/Guide.hpp        the tagged word
 *   ActiveScopeGuard       crest/runtime/Guide.hpp        the ATC pin
 *   Sama                   crest/runtime/Sama.h           the four arenas
 *   ObjectCollector        crest/runtime/ObjectCollector.h the migrator thread
 *   globalConfig.h                                        the fork handshake
 *
 * A "managed value" is a Redis string value whose payload lives in a block
 * allocated here and whose *slot* (robj->ptr) holds the OBASE guide word:
 *
 *   |63 MLOCK|62 ACC|61-60 HEAP|59-54 ATC|53 WRITE|52-48 CIW|47-0 ADDR|
 *
 * so robj->ptr is NOT a pointer while a value is managed. That is deliberate
 * -- server.h gives such a value the encoding OBJ_ENCODING_RAW_GUIDED, which
 * is not part of sdsEncodedObject(), so an unconverted call site takes the
 * "unknown encoding" branch (a panic or an assert) instead of dereferencing a
 * tagged word.
 *
 * Managed block layout (refinery_alloc_value):
 *
 *   +0   uint64_t sds_off     offset from the block base to the sds payload
 *   +8   sds header (5 B for SDS_TYPE_16)
 *   +.   payload, NUL
 *
 * The guide word holds the *block base*, never the sds payload, because the
 * collector sizes its copy with jem_sallocx(base) and memcpy()s from base;
 * an interior pointer would break both. The 8-byte prefix is what lets a
 * relocated block still find its sds header (whose size is only recoverable
 * from the flags byte that sits inside it).
 */

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* 1 when REFINERY=1 in the environment: the runtime is constructed and the
 * collector runs. 0 leaves every entry point below a no-op and Redis behaves
 * exactly as stock (the runtime is compiled in but inert). */
extern int refinery_on;

/* Read REFINERY* from the environment. Call before initServer(). Returns
 * refinery_on. Does not allocate the arenas. */
int refinery_boot(void);

/* Construct Sama (COLD + HOT + WRITE arenas) and start the collector thread.
 * Call from main() after initServer(). No-op when refinery_on == 0. */
void refinery_start(void);

/* OBASE runtime mode: 0 none, 1 decay (scan only), 2 migrate.
 * DEBUG REFINERY none|decay|migrate. */
void refinery_set_mode(int mode);
int refinery_get_mode(void);

/* ---- the guide word ------------------------------------------------- */
/* slot is &robj->ptr. Each returns the canonical address of the managed
 * block. deref() declares a read, deref_write() declares a write (sets the
 * WRITE bit, A1), deref_ro() reads the word with no atomic at all. Inside a
 * fork child every one of them behaves as deref_ro (invariant I3), which is
 * why the RDB child needs no special path of its own. */
uint64_t refinery_guide_deref(void **slot);
uint64_t refinery_guide_deref_write(void **slot);
uint64_t refinery_guide_deref_ro(void **slot);
/* Declare a write to the object in `slot` without needing its address: the
 * A1 write bit, set at the point Redis replaces or mutates a value. */
void refinery_mark_write(void **slot);

/* ---- managed blocks -------------------------------------------------- */
/* Allocate `n` bytes in the NEW heap (jem_malloc; the collector promotes it
 * later). Returns NULL if the runtime is off. */
void *refinery_alloc(size_t n);
/* Zero `slot`, register it in SODA and publish `p` in it. After this the slot
 * is a guide word. */
void refinery_install(void **slot, void *p);
/* Unregister `slot` and free the block it points at into its own arena.
 * Leaves the slot zeroed. */
void refinery_release(void **slot);
/* Unregister `slot` WITHOUT freeing the block: ownership has moved to another
 * slot. Used by the same-size overwrite path, which reuses the previous value's
 * block instead of allocating a new one. */
void refinery_disown(void **slot);
/* Usable size of a managed block (jem_sallocx). */
size_t refinery_block_size(const void *p);
/* 1 unless REFINERY_REDIS_REUSE=0: reuse the previous value's block on a
 * same-size overwrite (see refineryValueAdoptOver in refinery_obj.c). */
extern int refinery_reuse_blocks;
/* REFINERY_REDIS_ROBJ_REUSE=1 (default 0): on an overwrite keep the OLD robj
 * as the db value and write the new bytes into its block, instead of
 * installing the freshly-built argv robj. The value's *identity* becomes
 * stable, so a SET stops rewriting the 16 B robj header and stops rewriting
 * dictEntry.v.val -- the two unmanaged classes that carry 37 % of the COW at
 * under 1 % write utilization (D12 step 5). Requires refinery_reuse_blocks. */
extern int refinery_robj_reuse;
/* REFINERY_LRU_COPY=0 (default 1) drops `val->lru = old->lru` in dbSetValue.
 * The field feeds LRU/LFU eviction and OBJECT IDLETIME/FREQ only; under
 * maxmemory-policy noeviction nothing reads it. B1 of D12 step 6. */
extern int refinery_lru_copy;

/* Bytes currently held in managed blocks (jem_sallocx of every live block).
 * zmalloc_used_memory() adds this, so used_memory/maxmemory/MEMORY USAGE see
 * the whole value again. Returns 0 when the runtime is off. */
size_t refinery_managed_bytes(void);

/* Append a labelled /proc/self/smaps sample to $REFINERY_SNAP_DIR/smaps.txt,
 * in the same "=== REFINERY_SAMPLE label=<l> t_ms=<t> ===" framing that
 * crest/refinery/Snapshot.cc writes, so tools/thp-coverage.py reads both.
 * No-op when REFINERY_SNAP_DIR is unset. Works with REFINERY=0: the baseline
 * arm needs the AnonHugePages of Redis's own je_ heap. */
void refinery_smaps_sample(const char *label);

/* ---- scopes ---------------------------------------------------------- */
/* One pair per command (server.c call()) and one per cron pass. Nesting is
 * handled by ActiveScopeGuard's level counter. The scope must be SHORT: the
 * collector's signalMigrationBegin() waits for every thread with an open
 * scope to observe the new migration generation, which only happens at a
 * level-0 scope entry. */
void refinery_scope_enter(void);
void refinery_scope_exit(void);
/* Close every scope this thread still holds. SHUTDOWN reaches exit(0) from
 * inside call(), i.e. with the command's scope open; the ObjectCollector's
 * static destructor then joins the collector, which is itself waiting in
 * signalMigrationBegin() for that scope to be observed -- a deadlock that
 * leaves the server hung after "Redis is now ready to exit, bye bye". */
void refinery_scope_reset(void);
/* Terminate. With the runtime on this is _exit(), because the OBASE
 * ObjectCollector's migrator thread is a `while (true)` loop with no stop
 * condition and ~ObjectCollector joins it unconditionally
 * (crest/runtime/ObjectCollector.cc:70,171) -- so a process that constructed
 * the collector can never run its static destructors. Everything durable has
 * already happened in prepareForShutdown()/finishShutdown(): the RDB is
 * written, fsynced and renamed, the AOF is closed, the pidfile is unlinked.
 * With the runtime off this is a plain exit(), i.e. stock behaviour. */
void refinery_exit(int code);

/* ---- fork ------------------------------------------------------------ */
/* Parent, immediately before fork(): publish the fork intent and wait for any
 * migration round already in flight (I1). Opens the fork window (A2).
 *
 * Returns 0 if the caller may fork, -1 if the wait exceeded
 * REFINERY_FORK_WAIT_MAX_S (default 120) -- in which case the caller must NOT
 * fork, and the intent this call published has been retracted. Forking into a
 * live round lets migration move objects out from under a snapshot child,
 * which is exactly what I1 forbids: a deferred checkpoint is a service
 * degradation, a checkpoint taken across a migration is a correctness
 * failure. Always returns 0 when the runtime is off. */
int refinery_fork_begin(void);
/* REFINERY_FORK_WAIT_MAX_S, for the log line on the refusal path. */
unsigned refinery_fork_wait_max(void);
/* Child, immediately after fork(): read-only dereference from here on (I3). */
void refinery_fork_child(void);
/* Parent, when the child is reaped: close the window. */
void refinery_fork_end(void);

/* ---- introspection --------------------------------------------------- */
/* INFO refinery body. Returns bytes written (never more than buflen-1). */
size_t refinery_info(char *buf, size_t buflen);
/* HEAPS report (arena ranges + I1-I3 counters), for the M1 analysis. */
size_t refinery_heaps(char *buf, size_t buflen);
/* Walk SODA and write regions-slots.bin / regions-obj-h{0..3}.bin into
 * REFINERY_SNAP_DIR. One line of status into buf. */
size_t refinery_dump_regions(char *buf, size_t buflen);
/* Counters the host maintains (values adopted/released, bytes) so INFO can
 * report them next to the runtime's own. */
void refinery_note_adopt(size_t bytes);
void refinery_note_release(size_t bytes);

#ifdef __cplusplus
}
#endif

#endif /* REFINERY_GUIDE_H */
