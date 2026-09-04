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
    char *ohdr = (char *)sdsAllocPtr(s);
    size_t osz = sdsAllocSize(s);            /* header + alloc + NUL */
    size_t hdrlen = (size_t)((char *)s - ohdr);

    char *blk = refinery_alloc(REFINERY_BLK_PREFIX + osz);
    if (blk == NULL) return;                 /* stay unmanaged; never fail a write */
    *(uint64_t *)blk = (uint64_t)(REFINERY_BLK_PREFIX + hdrlen);
    memcpy(blk + REFINERY_BLK_PREFIX, ohdr, osz);

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

/* Re-adopt after an in-place mutation (APPEND, SETRANGE): the mutation itself
 * is the write, so this always marks. Separate name so the call sites read as a
 * matched pair with refineryValueUnguard(). */
void refineryValueReadopt(robj *o) {
    refineryValueAdoptWritten(o);
}
