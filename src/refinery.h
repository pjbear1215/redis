#ifndef REFINERY_H
#define REFINERY_H

/*
 * Refinery (Branch A) -- M1 instrumentation for stock Redis 7.2.
 *
 * Two instruments, both in one mmap(MAP_SHARED|MAP_ANONYMOUS) region:
 *
 *  1. a write log: (addr, seq, len, tag) records, the same 24-byte layout as
 *     crest/refinery/Instrument.h so the offline analysis reads both systems
 *     with one parser. Shared anonymous memory is not copy-on-write, so
 *     appending during the BGSAVE child's lifetime costs no COW page and
 *     cannot inflate the parent's copied-page count -- which is exactly the
 *     quantity M1 divides by.
 *
 *  2. a page-class map: one byte per 4 KiB page, a bitmask of the object
 *     kinds ever allocated in that page. Redis has no per-class arenas, so
 *     this is the Redis analog of OBASE's HEAPS/REGIONS report. A page whose
 *     byte has exactly one bit set is attributed to that class; a page with
 *     several is "mixed", which is itself the page-composition measurement.
 *
 * Everything is a no-op until refineryInit() succeeds, and everything is
 * compiled out with REFINERY_WRITELOG=0.
 */

#include <stdint.h>
#include <stddef.h>

/* --- write-log record tags ------------------------------------------- */
#define REFINERY_W_VALUE_NEW     0  /* payload of a freshly allocated value */
#define REFINERY_W_VALUE_INPLACE 1  /* memcpy/memset into an existing value */
#define REFINERY_W_DICT_VAL      2  /* dictEntry.v.val store                */
#define REFINERY_W_DICT_ENTRY    3  /* dictEntry allocated and filled       */
#define REFINERY_W_KEY_SDS       4  /* key sds (sdsdup at insert)           */
#define REFINERY_W_ROBJ_LRU      5  /* robj->lru store on lookup            */
#define REFINERY_W_ROBJ_NEW      6  /* robj header of a new value           */
#define REFINERY_W_DICT_TABLE    7  /* bucket array store / rehash          */
#define REFINERY_W_CHILD         8  /* written by the BGSAVE child          */
#define REFINERY_W_NTAG          9

/* --- page-class bits -------------------------------------------------- */
#define REFINERY_C_VALUE_SDS   0x01
#define REFINERY_C_VALUE_ROBJ  0x02
#define REFINERY_C_KEY_SDS     0x04
#define REFINERY_C_DICT_ENTRY  0x08
#define REFINERY_C_DICT_TABLE  0x10
#define REFINERY_C_OTHER       0x20
/* Refinery (Branch A), D12: a value payload living in a block the OBASE
 * runtime manages. Distinct from VALUE_SDS, which is the transient zmalloc'd
 * sds the network layer builds and refineryValueAdopt() copies out of. */
#define REFINERY_C_VALUE_MANAGED 0x40

#ifndef REFINERY_WRITELOG
#define REFINERY_WRITELOG 1
#endif

#if REFINERY_WRITELOG

/* Called once from main() before anything is allocated in the keyspace.
 * Region size in MiB from the environment variable REFINERY_REGION_MB
 * (default 2048). Set REFINERY_REGION_MB=0 to disable the instrument. */
void refineryInit(void);

/* Region address range, so the offline analysis can exclude it by address
 * as well as by the fact that it is MAP_SHARED. (0,0 when disabled.) */
void refineryRegionRange(uint64_t *base, uint64_t *end);

/* Append one record. Cheap: one bounds check and a 24-byte store. */
void refineryLogWrite(const void *addr, size_t len, int tag);

/* OR `bits` into the class byte of every page covered by [addr, addr+len). */
void refineryTagPages(const void *addr, size_t len, uint8_t bits);

/* Discard the log (not the page-class map) and start a new window.
 * Called automatically in the parent just before fork(). */
void refineryLogReset(void);

/* Switch appends to the child ring; called in the fork child. */
void refineryEnterChild(void);

/* Freeze the parent log at the moment the child is reaped, so the numerator
 * covers exactly the child's lifetime and not the traffic that follows it.
 * Called from checkChildrenDone(). */
void refineryMarkChildEnd(void);

/* Write writelog-parent.bin, writelog-child.bin, pageclass.bin and
 * refinery-meta.txt into `dir`. Returns 0 on success. */
int refineryDump(const char *dir);

/* One-line status for DEBUG REFINERY REGION / logs. Caller-provided buf. */
void refineryStatus(char *buf, size_t buflen);

/* Control arm for the F1 finding: when 1, Redis's LRU/LFU update on lookup
 * is NOT suppressed while a saving child is alive (db.c). Set from
 * REFINERY_LRU_IN_CHILD=1 in refineryInit(). Default 0 = stock behaviour. */
extern int refinery_lru_always;

#else /* !REFINERY_WRITELOG */

#define refineryInit()                       do {} while (0)
#define refineryRegionRange(b, e)            do { *(b) = 0; *(e) = 0; } while (0)
#define refineryLogWrite(a, l, t)            do {} while (0)
#define refineryTagPages(a, l, b)            do {} while (0)
#define refineryLogReset()                   do {} while (0)
#define refineryEnterChild()                 do {} while (0)
#define refineryMarkChildEnd()               do {} while (0)
#define refineryDump(d)                      (0)
#define refineryStatus(b, l)                 do { (b)[0] = 0; } while (0)
#define refinery_lru_always                  0

#endif /* REFINERY_WRITELOG */

#endif /* REFINERY_H */
