/* Refinery (Branch A) -- see refinery.h. libc only: this object is linked
 * into redis-server, redis-cli and redis-benchmark (dict.c and sds.c are
 * shared), so it must not reference any Redis symbol. */

#include "refinery.h"

#if REFINERY_WRITELOG

#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

#define REF_PAGE_SHIFT 12
#define REF_PAGE (1UL << REF_PAGE_SHIFT)
/* One radix leaf covers 1 GiB of address space = 262144 pages = 262144 B. */
#define REF_GRAN_SHIFT 30
#define REF_LEAF_PAGES (1UL << (REF_GRAN_SHIFT - REF_PAGE_SHIFT))
/* 256 TiB of user address space / 1 GiB (x86-64 user VA is 47 bits, so a
 * heap at 0x7f.... needs > 2^16 granules -- 2^16 silently dropped every
 * tag in the first smoke run). */
#define REF_TOP_ENTRIES (1UL << 18)

typedef struct {
    uint64_t addr;
    uint64_t seq;
    uint32_t len;
    uint32_t tag;
} refRec;

typedef struct {
    uint64_t capacity;   /* records per ring */
    uint64_t head;       /* records appended by the parent */
    uint64_t dropped;    /* records lost because the ring filled */
    uint64_t chead;      /* records appended by the child */
    uint64_t cdropped;
    uint64_t seq;
    uint64_t bump;       /* bump allocator cursor, offset into the region */
    uint64_t size;       /* region size */
    uint64_t mark;       /* parent records at child exit; 0 = still open */
    uint64_t leaf_off[REF_TOP_ENTRIES]; /* 0 = absent */
} refHdr;

static char *ref_base = NULL;
static uint64_t ref_size = 0;
static refHdr *ref_hdr = NULL;
static refRec *ref_parent_ring = NULL;
static refRec *ref_child_ring = NULL;
static int ref_in_child = 0;
int refinery_lru_always = 0;

/* Bump-allocate from the region. Not thread-safe by design: only the main
 * thread tags pages. Returns 0 when exhausted. */
static uint64_t refBump(uint64_t n) {
    uint64_t off;
    n = (n + 63) & ~63UL;
    if (ref_hdr == NULL || ref_hdr->bump + n > ref_hdr->size) return 0;
    off = ref_hdr->bump;
    ref_hdr->bump += n;
    return off;
}

void refineryInit(void) {
    const char *s;
    uint64_t mb = 2048, ring_bytes;

    s = getenv("REFINERY_LRU_IN_CHILD");
    if (s != NULL && strtoul(s, NULL, 10) != 0) refinery_lru_always = 1;

    if (ref_base != NULL) return;
    s = getenv("REFINERY_REGION_MB");
    if (s != NULL) mb = strtoull(s, NULL, 10);
    if (mb == 0) return;

    ref_size = mb << 20;
    ref_base = mmap(NULL, ref_size, PROT_READ | PROT_WRITE,
                    MAP_SHARED | MAP_ANONYMOUS, -1, 0);
    if (ref_base == MAP_FAILED) {
        fprintf(stderr, "refinery: mmap(%llu MiB) failed: %s\n",
                (unsigned long long)mb, strerror(errno));
        ref_base = NULL;
        ref_size = 0;
        return;
    }
    ref_hdr = (refHdr *)ref_base;
    memset(ref_hdr, 0, sizeof(*ref_hdr));
    ref_hdr->size = ref_size;
    ref_hdr->bump = (sizeof(refHdr) + 63) & ~63UL;

    /* Half the remaining region to the parent ring, an eighth to the child. */
    ring_bytes = (ref_size - ref_hdr->bump) / 2;
    ref_hdr->capacity = ring_bytes / sizeof(refRec);
    ref_parent_ring = (refRec *)(ref_base + refBump(ref_hdr->capacity * sizeof(refRec)));
    ref_child_ring = (refRec *)(ref_base + refBump((ref_hdr->capacity / 4) * sizeof(refRec)));

    fprintf(stderr, "refinery: region %llu MiB at %p-%p, %llu records/ring\n",
            (unsigned long long)mb, (void *)ref_base,
            (void *)(ref_base + ref_size),
            (unsigned long long)ref_hdr->capacity);
}

void refineryRegionRange(uint64_t *base, uint64_t *end) {
    *base = (uint64_t)(uintptr_t)ref_base;
    *end = ref_base ? (uint64_t)(uintptr_t)ref_base + ref_size : 0;
}

void refineryLogWrite(const void *addr, size_t len, int tag) {
    refRec *r;
    uint64_t idx;

    if (ref_hdr == NULL) return;
    if (!ref_in_child && ref_hdr->mark != 0) return; /* window closed */
    if (ref_in_child) {
        idx = ref_hdr->chead;
        if (idx >= ref_hdr->capacity / 4) { ref_hdr->cdropped++; return; }
        ref_hdr->chead = idx + 1;
        r = &ref_child_ring[idx];
    } else {
        idx = ref_hdr->head;
        if (idx >= ref_hdr->capacity) { ref_hdr->dropped++; return; }
        ref_hdr->head = idx + 1;
        r = &ref_parent_ring[idx];
    }
    r->addr = (uint64_t)(uintptr_t)addr;
    r->seq = ref_hdr->seq++;
    r->len = (uint32_t)len;
    r->tag = (uint32_t)tag;
}

void refineryTagPages(const void *addr, size_t len, uint8_t bits) {
    uint64_t a, end, gran, off;
    unsigned char *leaf;

    if (ref_hdr == NULL || addr == NULL || len == 0) return;
    a = (uint64_t)(uintptr_t)addr;
    end = a + len - 1;
    for (; a <= end; a = (a & ~(REF_PAGE - 1)) + REF_PAGE) {
        gran = a >> REF_GRAN_SHIFT;
        if (gran >= REF_TOP_ENTRIES) return;
        off = ref_hdr->leaf_off[gran];
        if (off == 0) {
            off = refBump(REF_LEAF_PAGES);
            if (off == 0) return;
            memset(ref_base + off, 0, REF_LEAF_PAGES);
            ref_hdr->leaf_off[gran] = off;
        }
        leaf = (unsigned char *)(ref_base + off);
        leaf[(a >> REF_PAGE_SHIFT) & (REF_LEAF_PAGES - 1)] |= bits;
    }
}

void refineryLogReset(void) {
    if (ref_hdr == NULL) return;
    ref_hdr->head = 0;
    ref_hdr->dropped = 0;
    ref_hdr->chead = 0;
    ref_hdr->cdropped = 0;
    ref_hdr->seq = 0;
    ref_hdr->mark = 0;
}

void refineryMarkChildEnd(void) {
    if (ref_hdr == NULL || ref_hdr->mark != 0) return;
    ref_hdr->mark = ref_hdr->head ? ref_hdr->head : (uint64_t)-1;
}

void refineryEnterChild(void) { ref_in_child = 1; }

static int refWriteFile(const char *dir, const char *name,
                        const void *buf, size_t len) {
    char path[1024];
    int fd;
    ssize_t n;
    const char *p = (const char *)buf;

    snprintf(path, sizeof(path), "%s/%s", dir, name);
    fd = open(path, O_CREAT | O_TRUNC | O_WRONLY, 0644);
    if (fd < 0) return -1;
    while (len > 0) {
        n = write(fd, p, len > (1u << 20) ? (1u << 20) : len);
        if (n <= 0) { close(fd); return -1; }
        p += n;
        len -= (size_t)n;
    }
    close(fd);
    return 0;
}

int refineryDump(const char *dir) {
    char path[1024];
    FILE *f;
    uint64_t g, pages = 0;

    uint64_t nparent;

    if (ref_hdr == NULL) return -1;
    nparent = ref_hdr->head;
    if (ref_hdr->mark != 0 && ref_hdr->mark != (uint64_t)-1 &&
        ref_hdr->mark < nparent)
        nparent = ref_hdr->mark;
    if (refWriteFile(dir, "writelog-parent.bin", ref_parent_ring,
                     nparent * sizeof(refRec)) != 0) return -1;
    if (refWriteFile(dir, "writelog-child.bin", ref_child_ring,
                     ref_hdr->chead * sizeof(refRec)) != 0) return -1;

    /* pageclass.bin: repeated (uint64 page_index, uint8 bits), only for
     * pages that carry at least one bit. */
    snprintf(path, sizeof(path), "%s/pageclass.bin", dir);
    f = fopen(path, "wb");
    if (f == NULL) return -1;
    for (g = 0; g < REF_TOP_ENTRIES; g++) {
        unsigned char *leaf;
        uint64_t i;
        if (ref_hdr->leaf_off[g] == 0) continue;
        leaf = (unsigned char *)(ref_base + ref_hdr->leaf_off[g]);
        for (i = 0; i < REF_LEAF_PAGES; i++) {
            uint64_t page;
            if (leaf[i] == 0) continue;
            page = (g << (REF_GRAN_SHIFT - REF_PAGE_SHIFT)) + i;
            fwrite(&page, sizeof(page), 1, f);
            fwrite(&leaf[i], 1, 1, f);
            pages++;
        }
    }
    fclose(f);

    snprintf(path, sizeof(path), "%s/refinery-meta.txt", dir);
    f = fopen(path, "w");
    if (f == NULL) return -1;
    fprintf(f, "region_base 0x%llx\n", (unsigned long long)(uintptr_t)ref_base);
    fprintf(f, "region_end 0x%llx\n",
            (unsigned long long)((uintptr_t)ref_base + ref_size));
    fprintf(f, "parent_records %llu\n", (unsigned long long)nparent);
    fprintf(f, "parent_head %llu\n", (unsigned long long)ref_hdr->head);
    fprintf(f, "parent_mark %llu\n", (unsigned long long)ref_hdr->mark);
    fprintf(f, "parent_dropped %llu\n", (unsigned long long)ref_hdr->dropped);
    fprintf(f, "child_records %llu\n", (unsigned long long)ref_hdr->chead);
    fprintf(f, "child_dropped %llu\n", (unsigned long long)ref_hdr->cdropped);
    fprintf(f, "classified_pages %llu\n", (unsigned long long)pages);
    fclose(f);
    return 0;
}

void refineryStatus(char *buf, size_t buflen) {
    if (ref_hdr == NULL) {
        snprintf(buf, buflen, "refinery disabled");
        return;
    }
    snprintf(buf, buflen,
             "base=0x%llx end=0x%llx parent=%llu dropped=%llu child=%llu "
             "cdropped=%llu mark=%llu bump=%llu/%llu",
             (unsigned long long)(uintptr_t)ref_base,
             (unsigned long long)((uintptr_t)ref_base + ref_size),
             (unsigned long long)ref_hdr->head,
             (unsigned long long)ref_hdr->dropped,
             (unsigned long long)ref_hdr->chead,
             (unsigned long long)ref_hdr->cdropped,
             (unsigned long long)ref_hdr->mark,
             (unsigned long long)ref_hdr->bump,
             (unsigned long long)ref_hdr->size);
}

#endif /* REFINERY_WRITELOG */
