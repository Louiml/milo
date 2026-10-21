// Run-time dispatch to the best kernels this machine can run.
#include "tensor/kernels.hpp"

namespace milo::tensor::kernels {

// Defined in the per-instruction-set translation units.
const Table& scalarTable();

#if defined(MILO_X86)
const Table& avx2Table();
#endif

#if defined(MILO_ARM)
const Table& neonTable();
#endif

namespace {

bool detectAvx2() {
#if defined(MILO_X86)
#if defined(_MSC_VER)
  int info[4] = {0, 0, 0, 0};

  __cpuid(info, 0);
  const int maxLeaf = info[0];
  if (maxLeaf < 7) return false;

  // The OS has to have enabled the wide registers.
  __cpuid(info, 1);
  if ((info[2] & (1 << 27)) == 0) return false;  // OSXSAVE
  if ((info[2] & (1 << 28)) == 0) return false;  // AVX

  unsigned long long xcr0 = 0;
  _asm_xgetbv(0, &xcr0);
  if ((xcr0 & 0x6) != 0x6) return false;  // XMM and YMM state saved

  __cpuidex(info, 7, 0);
  return (info[1] & (1 << 5)) != 0;  // AVX2
#else
  __builtin_cpu_init();
  return __builtin_cpu_supports("avx2") != 0;
#endif
#else
  return false;
#endif
}

bool detectNeon() {
#if defined(MILO_ARM)
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
  return true;
#else
  return false;
#endif
#else
  return false;
#endif
}

}  // namespace

const char* isaName(Isa isa) {
  switch (isa) {
    case Isa::Scalar:
      return "scalar";
    case Isa::Avx2:
      return "avx2";
    case Isa::Neon:
      return "neon";
  }
  return "scalar";
}

bool hasAvx2() { return detectAvx2(); }
bool hasNeon() { return detectNeon(); }

Isa activeIsa() {
#if defined(MILO_X86)
  if (detectAvx2()) return Isa::Avx2;
#endif
#if defined(MILO_ARM)
  if (detectNeon()) return Isa::Neon;
#endif
  return Isa::Scalar;
}

const Table& table() {
#if defined(MILO_X86)
  static const Table& chosen = detectAvx2() ? avx2Table() : scalarTable();
  return chosen;
#elif defined(MILO_ARM)
  static const Table& chosen = detectNeon() ? neonTable() : scalarTable();
  return chosen;
#else
  static const Table& chosen = scalarTable();
  return chosen;
#endif
}

}  // namespace milo::tensor::kernels
