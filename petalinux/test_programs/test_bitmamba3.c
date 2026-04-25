/*
 * test_bitmamba3.c — Host driver for BitMamba-3 FPGA accelerator on Zybo Z7-20.
 *
 * Targets the top_bitmamba3_block.v wrapper integrated through the prior
 * project's axi_lite_csr (G:\Xilinx\rtl\cores\attention_engine\axi_lite_csr.sv)
 * + axi_hp_dma. Mirrors the structure of the existing test_dot_hp.c.
 *
 * Usage:
 *   test_bitmamba3 <input.bin> <weights.bin> <output.bin>
 *
 * This program:
 *   1. Loads activation tensor and weight blob from disk
 *   2. mmaps PS-side DDR (uncached) and copies the data to DDR-mapped buffers
 *   3. mmaps AXI-Lite control space at 0x43C0_0000
 *   4. Programs base addresses and starts the accelerator
 *   5. Polls status register until done
 *   6. Reads back output and writes to disk
 *   7. Reports cycle count and bandwidth
 *
 * Build (Yocto / PetaLinux):
 *   $ ${CC} -O2 -Wall -o test_bitmamba3 test_bitmamba3.c
 */

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <errno.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <time.h>

/* -------- AXI-Lite CSR map (mirrors axi_lite_csr.sv adapted for BitMamba-3) -------- */
#define BM3_CTRL_BASE          0x43C00000
#define BM3_REG_SIZE           0x1000

#define REG_CTRL               0x00  /* [0]=start [1]=mode_eval [31:2]=reserved */
#define REG_SEQLEN             0x04
#define REG_WEIGHT_BASE        0x08  /* DDR phys addr of ternary-packed weights */
#define REG_ACT_BASE           0x0C  /* DDR phys addr of input activation buffer */
#define REG_OUT_BASE           0x10  /* DDR phys addr for output hidden state */
#define REG_LAYER_IDX          0x14  /* current layer index */
#define REG_STATUS             0x18  /* [0]=busy [1]=done [2]=error */
#define REG_PERF_CYCLES        0x1C
#define REG_PERF_RD_BURSTS     0x20
#define REG_PERF_WR_BURSTS     0x24

/* -------- DDR3 reserved range for accelerator buffers (uncached via /dev/mem) -------- */
/*  In PetaLinux, reserve via device tree:
 *    bitmamba3_dma: bitmamba3_buf@30000000 {
 *        no-map;
 *        reg = <0x30000000 0x10000000>;  // 256 MB at 0x30000000
 *    };
 */
#define DMA_BUF_PHYS           0x30000000
#define DMA_BUF_SIZE           (256u << 20)  /* 256 MB */

#define WEIGHT_OFFSET          0x00000000   /* up to 192 MB ternary-packed */
#define ACT_OFFSET             0x0C000000   /* 64 MB activation scratch */
#define OUT_OFFSET             0x0F800000   /* 8 MB output buffer */

static volatile uint32_t *bm3_csr;
static uint8_t           *dma_buf;

static int mmap_dev(off_t phys, size_t size, void **vaddr, int writable)
{
    int fd = open("/dev/mem", O_RDWR | O_SYNC);
    if (fd < 0) { perror("open /dev/mem"); return -1; }
    int prot = PROT_READ | (writable ? PROT_WRITE : 0);
    void *p = mmap(NULL, size, prot, MAP_SHARED, fd, phys);
    close(fd);
    if (p == MAP_FAILED) { perror("mmap"); return -1; }
    *vaddr = p;
    return 0;
}

static int load_file(const char *path, void *dst, size_t max_bytes, size_t *out_bytes)
{
    int fd = open(path, O_RDONLY);
    if (fd < 0) { perror(path); return -1; }
    struct stat st;
    if (fstat(fd, &st) < 0) { perror("fstat"); close(fd); return -1; }
    if ((size_t)st.st_size > max_bytes) {
        fprintf(stderr, "%s too large: %zd > %zu\n", path, (ssize_t)st.st_size, max_bytes);
        close(fd);
        return -1;
    }
    ssize_t n = read(fd, dst, st.st_size);
    close(fd);
    if (n != st.st_size) { perror("read"); return -1; }
    *out_bytes = (size_t)st.st_size;
    return 0;
}

static int save_file(const char *path, const void *src, size_t bytes)
{
    int fd = open(path, O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd < 0) { perror(path); return -1; }
    ssize_t n = write(fd, src, bytes);
    close(fd);
    return (n == (ssize_t)bytes) ? 0 : -1;
}

static double now_sec(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main(int argc, char **argv)
{
    if (argc != 4) {
        fprintf(stderr, "usage: %s <input.bin> <weights.bin> <output.bin>\n", argv[0]);
        return 1;
    }

    /* Map AXI-Lite CSR */
    void *csr_v;
    if (mmap_dev(BM3_CTRL_BASE, BM3_REG_SIZE, &csr_v, 1) < 0) return 1;
    bm3_csr = (volatile uint32_t *)csr_v;

    /* Map DDR scratch (uncached) */
    void *dma_v;
    if (mmap_dev(DMA_BUF_PHYS, DMA_BUF_SIZE, &dma_v, 1) < 0) return 1;
    dma_buf = (uint8_t *)dma_v;

    /* Load weights and activations into DDR scratch */
    size_t weight_bytes, act_bytes;
    if (load_file(argv[2], dma_buf + WEIGHT_OFFSET,
                  ACT_OFFSET - WEIGHT_OFFSET, &weight_bytes) < 0) return 1;
    if (load_file(argv[1], dma_buf + ACT_OFFSET,
                  OUT_OFFSET - ACT_OFFSET, &act_bytes) < 0) return 1;
    fprintf(stderr, "loaded weights=%zu B, activations=%zu B\n",
            weight_bytes, act_bytes);

    /* Program control registers */
    bm3_csr[REG_WEIGHT_BASE / 4] = DMA_BUF_PHYS + WEIGHT_OFFSET;
    bm3_csr[REG_ACT_BASE   / 4] = DMA_BUF_PHYS + ACT_OFFSET;
    bm3_csr[REG_OUT_BASE   / 4] = DMA_BUF_PHYS + OUT_OFFSET;
    bm3_csr[REG_SEQLEN     / 4] = 2048;     /* default seqlen */
    bm3_csr[REG_LAYER_IDX  / 4] = 0;

    /* Kick off */
    double t0 = now_sec();
    bm3_csr[REG_CTRL / 4] = 0x1;   /* start */

    /* Poll for done */
    uint32_t status;
    int spin = 0;
    while (1) {
        status = bm3_csr[REG_STATUS / 4];
        if (status & 0x4) {
            fprintf(stderr, "ERROR bit set, status=0x%x\n", status);
            return 2;
        }
        if (status & 0x2) break;
        if (++spin > 100000000) {
            fprintf(stderr, "TIMEOUT after %d polls, status=0x%x\n", spin, status);
            return 3;
        }
        /* lightweight backoff */
        if ((spin & 0xFFFF) == 0) usleep(1);
    }
    double t1 = now_sec();

    /* Read perf counters */
    uint32_t cycles = bm3_csr[REG_PERF_CYCLES / 4];
    uint32_t rb     = bm3_csr[REG_PERF_RD_BURSTS / 4];
    uint32_t wb     = bm3_csr[REG_PERF_WR_BURSTS / 4];
    fprintf(stderr, "done in %.3f ms (HW cycles=%u, RD bursts=%u, WR bursts=%u)\n",
            (t1 - t0) * 1e3, cycles, rb, wb);

    /* Save output */
    size_t out_bytes = OUT_OFFSET ? 0x800000 : 0;  /* 8 MB max */
    /* Actual size depends on FSM-determined hidden state shape; simplified here. */
    if (save_file(argv[3], dma_buf + OUT_OFFSET, out_bytes) < 0) return 1;

    munmap(dma_v, DMA_BUF_SIZE);
    munmap(csr_v, BM3_REG_SIZE);
    return 0;
}
