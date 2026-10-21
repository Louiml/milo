/* =====================================================================
 * milo.h — the stable C ABI for the Milo runtime.
 *
 * This is the ONLY public surface of the Milo project. It is:
 *
 *   - plain C (compiles as C99 and as C++),
 *   - exception-free (no C++ exception ever crosses this boundary),
 *   - ABI-stable for the whole 0.x series.
 *
 * Everything else in the repository (src/**, include/milo/*.hpp) is
 * internal implementation and may change without notice.
 *
 * Language bindings (the Rust package manager `mpm`, the CPython
 * extension module, and third-party native extensions) must include this
 * header and nothing else from the project.
 *
 * Threading:
 *   - a milo_runtime* is NOT thread-safe. Use one per thread, or
 *     serialise access externally.
 *   - milo_last_error() refers to the calling thread only.
 * ===================================================================== */

#ifndef MILO_H
#define MILO_H

#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32) && defined(MILO_BUILD_SHARED)
#  if defined(MILO_BUILDING)
#    define MILO_API __declspec(dllexport)
#  else
#    define MILO_API __declspec(dllimport)
#  endif
#elif defined(__GNUC__) && __GNUC__ >= 4
#  define MILO_API __attribute__((visibility("default")))
#else
#  define MILO_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* ------------------------------------------------------------------ */
/* Versioning                                                          */
/* ------------------------------------------------------------------ */

/** Bumped whenever the layout or meaning of this header changes. */
#define MILO_ABI_VERSION 1u

/** The language version this runtime implements (see docs/spec). */
#define MILO_LANGUAGE_VERSION "0.1"

/** Runtime implementation version, matching the VERSION file. */
#define MILO_RUNTIME_VERSION "0.1.0"

MILO_API uint32_t milo_abi_version(void);
MILO_API const char* milo_version_string(void);
MILO_API const char* milo_language_version(void);

/* ------------------------------------------------------------------ */
/* Status codes                                                        */
/* ------------------------------------------------------------------ */

typedef enum milo_status {
  MILO_OK = 0,
  MILO_ERR_SYNTAX = 1,     /* the program could not be parsed          */
  MILO_ERR_TYPE = 2,       /* static type error                        */
  MILO_ERR_RUNTIME = 3,    /* error while executing                    */
  MILO_ERR_IO = 4,         /* file / device error                      */
  MILO_ERR_NOT_FOUND = 5,  /* unknown name, missing file, ...          */
  MILO_ERR_UNSUPPORTED = 6,
  MILO_ERR_NOMEM = 7,
  MILO_ERR_INTERNAL = 8
} milo_status;

/** Human-readable name of a status code. Never NULL. */
MILO_API const char* milo_status_to_string(milo_status status);

/* ------------------------------------------------------------------ */
/* Per-thread error detail                                             */
/* ------------------------------------------------------------------ */

/**
 * Copy the calling thread's last error message into `buf`.
 * Returns the number of bytes written (excluding the NUL), or the number
 * of bytes required if `buf` is too small (in which case the message is
 * truncated but always NUL-terminated).
 */
MILO_API size_t milo_last_error(char* buf, size_t cap);

/** Clear the calling thread's last error message. */
MILO_API void milo_clear_error(void);

/* ------------------------------------------------------------------ */
/* Runtime handle                                                       */
/* ------------------------------------------------------------------ */

typedef struct milo_runtime milo_runtime;

/** Create a runtime instance. Returns NULL on failure. */
MILO_API milo_runtime* milo_runtime_create(void);

/** Destroy a runtime instance. Passing NULL is a no-op. */
MILO_API void milo_runtime_destroy(milo_runtime* rt);

/* ------------------------------------------------------------------ */
/* Values                                                               */
/* ------------------------------------------------------------------ */

typedef struct milo_value milo_value;

typedef enum milo_value_kind {
  MILO_VALUE_NULL = 0,
  MILO_VALUE_BOOL = 1,
  MILO_VALUE_INT = 2,
  MILO_VALUE_FLOAT = 3,
  MILO_VALUE_STRING = 4,
  MILO_VALUE_LIST = 5,
  MILO_VALUE_DICT = 6,
  MILO_VALUE_TENSOR = 7,
  MILO_VALUE_MODEL = 8,
  MILO_VALUE_DATASET = 9,
  MILO_VALUE_MODULE = 10,
  MILO_VALUE_FUNCTION = 11,
  MILO_VALUE_BUILTIN = 12
} milo_value_kind;

/** A handle to a runtime value. Owned by the runtime. */
MILO_API milo_value_kind milo_value_kind_of(const milo_value* v);
MILO_API int milo_value_is_null(const milo_value* v);
MILO_API int milo_value_as_bool(const milo_value* v);
MILO_API int64_t milo_value_as_int(const milo_value* v);
MILO_API double milo_value_as_float(const milo_value* v);

/** Borrowed pointer to the string payload; NULL if `v` is not a string. */
MILO_API const char* milo_value_as_string(const milo_value* v);

/** Length in bytes of the string payload; 0 for non-strings. */
MILO_API size_t milo_value_string_length(const milo_value* v);

/** The element types a tensor can hold. */
typedef enum milo_dtype {
  MILO_DTYPE_F32 = 0,
  MILO_DTYPE_F16 = 1,
  MILO_DTYPE_BF16 = 2,
  MILO_DTYPE_F64 = 3,
  MILO_DTYPE_I8 = 4,
  MILO_DTYPE_I32 = 5,
  MILO_DTYPE_I64 = 6,
  MILO_DTYPE_U8 = 7,
  MILO_DTYPE_BOOL = 8
} milo_dtype;

/** Number of axes in a tensor value; 0 for other kinds. */
MILO_API int milo_value_tensor_rank(const milo_value* v);

/** Length of one axis; 0 for other kinds or for an axis out of range. */
MILO_API int64_t milo_value_tensor_dim(const milo_value* v, int axis);

/** Element type of a tensor; MILO_DTYPE_F32 for other kinds. */
MILO_API milo_dtype milo_value_tensor_dtype(const milo_value* v);

/**
 * Pointer to the tensor's data, or NULL for other kinds.
 *
 * The pointer stays valid while the value handle is alive. Only tensors
 * with contiguous storage are described this way.
 */
MILO_API const void* milo_value_tensor_data(const milo_value* v);

/** Size of the tensor's data in bytes; 0 for other kinds. */
MILO_API size_t milo_value_tensor_bytes(const milo_value* v);

/**
 * May the tensor's data be written to?
 *
 * Returns 1 when the memory can be changed, 0 when it is read-only — a
 * memory-mapped file, or a tensor that belongs to somebody else.
 */
MILO_API int milo_value_tensor_writable(const milo_value* v);

/** Release a value handle obtained from the runtime. NULL is a no-op. */
MILO_API void milo_value_release(milo_value* v);

/* ------------------------------------------------------------------ */
/* Running programs                                                     */
/* ------------------------------------------------------------------ */

/**
 * Parse and statically check `src` without executing it.
 * On failure the thread's error message is set.
 */
MILO_API milo_status milo_runtime_check(milo_runtime* rt, const char* src,
                                        size_t len, const char* filename);

/**
 * Parse, check, and execute `src`.
 *
 * On success, and if `out_value` is non-NULL, a value handle is stored in
 * `*out_value`. The handle must be released with milo_value_release().
 *
 * On failure the thread's error message is set and MILO_ERR_* is returned.
 */
MILO_API milo_status milo_runtime_run(milo_runtime* rt, const char* src,
                                       size_t len, const char* filename,
                                       milo_value** out_value);

/**
 * Convenience wrapper around milo_runtime_run() that prints the result to
 * stdout using Milo's own printing rules.
 */
MILO_API milo_status milo_runtime_run_and_print(milo_runtime* rt, const char* src,
                                                 size_t len, const char* filename);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* MILO_H */
