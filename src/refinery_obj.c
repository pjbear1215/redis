/*
 * Refinery (Branch A), D12 -- Redis-side glue for managed string values.
 *
 * A *managed value* is a db value with type OBJ_STRING whose payload lives in
 * a block the OBASE runtime can relocate, and whose robj->ptr holds the guide
 * word rather than an sds pointer. Its encoding is OBJ_ENCODING_RAW_GUIDED,
 * which deliberately is NOT part of sdsEncodedObject(): a call site that was
 * not converted takes the "unknown string encoding" branch (a panic or an
 * assert) instead of dereferencing a tagged word as a pointer. That is the
 * design's single most important safety device (DESIGN.md e.1).
 *
 * Block layout (see refinery_guide.h): an 8-byte prefix holding the offset of
 * the sds payload, then the sds allocation copied verbatim. The guide word
 * holds the BLOCK BASE, because the collector sizes its copy with
 * jem_sallocx(base) and memcpy()s from base; the sds payload is an interior
 * pointer (the sds header sits before it) and would break both. The prefix is
 * what lets a relocated block find its sds again -- the header's own size is
 * only recoverable from the flags byte that lives inside the header.
 *
 * Keys, dict entries and the bucket array are never managed (D12 scope:
 * "string values only, keys and dict pinned").
 */

#include "server.h"
#include "refinery_guide.h"

/* Bytes reserved at the base of a managed block for the sds offset. 8 keeps
 * the sds header 8-aligned and, at 256 B values, keeps the block in the same
 * jemalloc size class (270 -> 320, as before). */
#define REFINERY_BLK_PREFIX 8

/* Bytes a managed block needs to hold `s` COMPACTLY: the prefix, the sds
 * header, the string and its NUL -- and nothing else.
 *
 * Copying the source sds *allocation* instead (sdsAllocSize) is what the first
 * version did, and it cost a whole jemalloc size class. sdsnewlen() asks for
 * hdr+len+1 = 262 B for a 256 B value, jemalloc gives it the 320 B class, and
 * sds records alloc = 320-5-1 = 314; sdsAllocSize then reports 320, so the
 * block asked for 8+320 = 328 and landed in the 384 B class. Every managed
 * value was 20 % larger than the stock one it replaced -- 960 MB on the 15 M
 * key E2 dataset -- and, worse for the measurement, it spread the same objects
 * over 20 % more pages than the baseline arm they are compared against.
 *
 * Rebuilding the sds compactly (alloc == len) puts 8+5+256+1 = 270 back in the
 * 320 B class, i.e. exactly the stock footprint. sdsavail() is then 0, which
 * is correct: a managed value never grows in place -- APPEND and SETRANGE
 * unguard to a zmalloc'd copy first. */
static inline size_t refineryCompactSize(sds s, size_t *hdrlen_out) {
    size_t hdrlen = (size_t)((char *)s - (char *)sdsAllocPtr(s));
    if (hdrlen_out) *hdrlen_out = hdrlen;
    return REFINERY_BLK_PREFIX + hdrlen + sdslen(s) + 1;
}

/* Write `s` into `blk` compactly and publish the offset of its payload. */
static inline void refineryWriteBlock(char *blk, sds s, size_t hdrlen) {
    size_t len = sdslen(s);
    *(uint64_t *)blk = (uint64_t)(REFINERY_BLK_PREFIX + hdrlen);
    memcpy(blk + REFINERY_BLK_PREFIX, (char *)s - hdrlen, hdrlen);
    memcpy(blk + REFINERY_BLK_PREFIX + hdrlen, s, len + 1);
    sdssetalloc((sds)(blk + REFINERY_BLK_PREFIX + hdrlen), len);
}

static inline sds refinerySdsFromBlock(uint64_t base) {
    char *b = (char *)(uintptr_t)base;
    return (sds)(b + *(uint64_t *)b);
}

sds refineryValueSds(robj *o) {
    return refinerySdsFromBlock(refinery_guide_deref(&o->ptr));
}

sds refineryValueSdsW(robj *o) {
    return refinerySdsFromBlock(refinery_guide_deref_write(&o->ptr));
}

sds refineryValueSdsRO(robj *o) {
    return refinerySdsFromBlock(refinery_guide_deref_ro(&o->ptr));
}

/* RAW -> RAW_GUIDED. Copies the whole sds allocation into a managed block and
 * frees the original. Called at the three install points (dbAddInternal,
 * dbSetValue, dbAddRDBLoad); everything else in Redis keeps building values
 * exactly as before, which is what makes the query-buffer donation path
 * (networking.c PROTO_MBULK_BIG_ARG) harmless: the robj owns that sds by then,
 * so freeing it here is correct. */
void refineryValueAdopt(robj *o) {
    if (!refinery_on) return;
    if (o->type != OBJ_STRING || o->encoding != OBJ_ENCODING_RAW) return;
    if (o->refcount >= OBJ_FIRST_SPECIAL_REFCOUNT) return;

    sds s = o->ptr;
    size_t hdrlen;
    size_t need = refineryCompactSize(s, &hdrlen);
    size_t osz = need - REFINERY_BLK_PREFIX;

    char *blk = refinery_alloc(need);
    if (blk == NULL) return;                 /* stay unmanaged; never fail a write */
    refineryWriteBlock(blk, s, hdrlen);
    /* Refinery (Branch A), D12: this memcpy IS the value write once the runtime
     * is on -- the write the network layer already logged went into the
     * transient zmalloc'd sds this copies out of, which is freed immediately.
     * Without this record the M1 numerator would lose the payload term
     * entirely and the managed heaps would read as pure waste. */
    refineryTagPages(blk, need, REFINERY_C_VALUE_MANAGED);
    refineryLogWrite(blk + REFINERY_BLK_PREFIX, osz, REFINERY_W_VALUE_NEW);

    sdsfree(s);
    o->ptr = NULL;                           /* the word must be clean metadata */
    refinery_install(&o->ptr, blk);
    o->encoding = OBJ_ENCODING_RAW_GUIDED;
    refinery_note_adopt(osz);
}

/* RAW_GUIDED -> RAW, payload back in zmalloc'd memory. Used as the prologue of
 * the in-place mutation commands (APPEND, SETRANGE): sds growth reallocates
 * through zrealloc, which a managed block cannot survive, so those commands
 * unguard, run stock, and re-adopt. Costs two copies per mutation; the SET/GET
 * workload never takes this path. */
void refineryValueUnguard(robj *o) {
    if (o->encoding != OBJ_ENCODING_RAW_GUIDED) return;
    sds s = refineryValueSds(o);
    size_t osz = sdsAllocSize(s);
    sds copy = sdsnewlen(s, sdslen(s));
    refinery_release(&o->ptr);
    o->ptr = copy;
    o->encoding = OBJ_ENCODING_RAW;
    refinery_note_release(osz);
}

/* Free path (freeStringObject). Leaves the object as an empty RAW string so
 * the caller's sdsfree(o->ptr) is a no-op. */
void refineryValueRelease(robj *o) {
    if (o->encoding != OBJ_ENCODING_RAW_GUIDED) return;
    sds s = refineryValueSds(o);             /* pins, and vetoes a live migration */
    size_t osz = sdsAllocSize(s);
    refinery_release(&o->ptr);
    o->encoding = OBJ_ENCODING_RAW;
    o->ptr = NULL;
    refinery_note_release(osz);
}

/* Adopt and declare the value written (A1's write axis).
 *
 * Which installs count as writes is the whole content of the write axis on
 * Redis. F2 (DESIGN.md 0) says a SET over an existing key is alloc-new-then-
 * free: the *new* object is the write. So dbSetValue() marks, dbAdd() on a key
 * that did not exist does not, and neither does an RDB load. A value that is
 * only ever created and read therefore reaches HOT_HEAP, and one that is
 * replaced reaches WRITE_HEAP -- which is exactly the segregation a checkpoint
 * fork needs, since a page of replaced values is a page the parent will dirty
 * again. The bit is sticky and the collector's epoch clear (every
 * REFINERY_WRITE_EPOCH_K windows) is the hysteresis. */
void refineryValueAdoptWritten(robj *o) {
    refineryValueAdopt(o);
    if (objIsGuided(o)) refinery_mark_write(&o->ptr);
}

/* Overwrite of an existing key: reuse the previous value's managed block when
 * it is big enough.
 *
 * This is what keeps the integration faithful to F2 (DESIGN.md 0). Stock Redis
 * SET is alloc-new-then-free and the freed 320 B region is handed straight back
 * to the next SET, so a steady SET stream writes onto pages that were already
 * resident -- and therefore shared -- at fork time. Allocating a fresh block per
 * overwrite instead would quietly convert Redis into an alloc-new system whose
 * value writes never land on shared pages, which would make the COW measurement
 * a measurement of the allocator change rather than of the placement policy.
 * Reusing the block reproduces the original behaviour inside the managed heap,
 * and it is also what gives the write axis something to classify: the block
 * keeps its identity, its guide word and its WRITE bit across overwrites.
 *
 * Only safe when the old value is unshared (refcount 1): dbSetValue() frees it
 * immediately afterwards, but another holder (slowlog, a Lua argv cache entry)
 * would be left reading a block that now belongs to the new value. The old slot
 * is disowned BEFORE the new one is installed, so at no instant do two
 * registered guide slots point at the same block.
 *
 * REFINERY_REDIS_REUSE=0 turns it off, which is the alloc-new contrast arm.
 */
void refineryValueAdoptOver(robj *val, robj *old) {
    if (refinery_on && refinery_reuse_blocks && old != NULL && old != val &&
        old->refcount == 1 && objIsGuided(old) &&
        val->type == OBJ_STRING && val->encoding == OBJ_ENCODING_RAW &&
        val->refcount < OBJ_FIRST_SPECIAL_REFCOUNT)
    {
        sds ns = val->ptr;
        size_t nhdr;
        size_t need = refineryCompactSize(ns, &nhdr);
        char *blk = (char *)(uintptr_t)refinery_guide_deref_write(&old->ptr);
        if (blk && refinery_block_size(blk) >= need) {
            refineryWriteBlock(blk, ns, nhdr);
            refineryLogWrite(blk + REFINERY_BLK_PREFIX, need - REFINERY_BLK_PREFIX,
                             REFINERY_W_VALUE_INPLACE);
            sdsfree(ns);
            refinery_disown(&old->ptr);
            old->encoding = OBJ_ENCODING_RAW;
            old->ptr = NULL;
            val->ptr = NULL;
            refinery_install(&val->ptr, blk);
            val->encoding = OBJ_ENCODING_RAW_GUIDED;
            refinery_mark_write(&val->ptr);
            return;
        }
    }
    refineryValueAdoptWritten(val);
}

/* Overwrite that keeps the OLD robj as the db value (D12 step 6, B2).
 *
 * refineryValueAdoptOver above already reuses the old value's *block*, so the
 * payload write lands where stock Redis's would. What it cannot avoid is that
 * Redis replaces the value's *identity* on every SET: the network layer builds
 * a fresh 16 B robj, dbSetValue installs it with dictSetVal() and frees the
 * old one. Two consequences, both measured in step 5 and both invisible in the
 * numerator:
 *
 *  - the new robj is a fresh 16 B allocation. jemalloc's tcache is LIFO, so
 *    it lands at the address the PREVIOUS SET freed -- a uniformly scattered
 *    address in the 234 MiB robj region -- and createObject() writes all 16
 *    bytes there. A few hundred thousand SETs therefore dirty essentially
 *    every robj page: 50,865 pages, 199 MiB, at 4.2 % write utilization.
 *  - dictSetVal() writes 8 bytes into dictEntry.v.val, in a 24 B slab shared
 *    with ~170 keys and entries per page: 62,491 pages at 0.32 %.
 *
 * Together that is 37 % of the write-aware arm's COW carrying under 1 % of the
 * writes -- the "unmanaged half". Neither is reachable by placing the payload
 * better, because neither is the payload. What removes both is making the
 * value's identity stable: keep `old`, write the new bytes into its block, and
 * never touch the robj or the dict entry.
 *
 * Returns `old` when it took over, NULL when the caller must proceed normally.
 * Preconditions: `old` is managed and unshared (refcount 1 -- another holder,
 * a slowlog entry or a Lua argv cache entry, would see the value change under
 * it), `val` is a plain RAW string the caller owns, and old's block is big
 * enough. The caller must NOT install `val` and must NOT free `old`.
 */
robj *refineryValueOverwriteInPlace(robj *val, robj *old) {
    if (!refinery_on || !refinery_robj_reuse) return NULL;
    if (old == NULL || old == val || !objIsGuided(old)) return NULL;
    if (old->refcount != 1) return NULL;
    if (val->type != OBJ_STRING || val->encoding != OBJ_ENCODING_RAW) return NULL;
    if (val->refcount >= OBJ_FIRST_SPECIAL_REFCOUNT) return NULL;

    sds ns = val->ptr;
    size_t nhdr;
    size_t need = refineryCompactSize(ns, &nhdr);
    /* deref_write, not deref: this IS the write, and it is what puts the
     * object in WRITE_HEAP (A1). */
    char *blk = (char *)(uintptr_t)refinery_guide_deref_write(&old->ptr);
    if (blk == NULL || refinery_block_size(blk) < need)
        return NULL;

    refineryWriteBlock(blk, ns, nhdr);
    refineryLogWrite(blk + REFINERY_BLK_PREFIX, need - REFINERY_BLK_PREFIX,
                     REFINERY_W_VALUE_INPLACE);
    return old;
}

/* Re-adopt after an in-place mutation (APPEND, SETRANGE): the mutation itself
 * is the write, so this always marks. Separate name so the call sites read as a
 * matched pair with refineryValueUnguard(). */
void refineryValueReadopt(robj *o) {
    refineryValueAdoptWritten(o);
}
