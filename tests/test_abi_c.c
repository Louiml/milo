/* A C program that includes the public header.
 *
 * If this compiles and links, the header is genuinely C-clean and the
 * ABI works from a foreign language. This is the test that stops the
 * header from quietly acquiring a C++ dependency.
 */
#include "milo/milo.h"

#include <stdio.h>
#include <string.h>

static int failures = 0;

#define CHECK(condition)                                              \
  do {                                                                \
    if (!(condition)) {                                               \
      fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #condition); \
      ++failures;                                                     \
    }                                                                 \
  } while (0)

int main(void) {
  CHECK(milo_abi_version() == MILO_ABI_VERSION);
  CHECK(milo_version_string() != NULL);
  CHECK(milo_language_version() != NULL);
  CHECK(strcmp(milo_status_to_string(MILO_OK), "ok") == 0);
  CHECK(strcmp(milo_status_to_string(MILO_ERR_SYNTAX), "syntax error") == 0);

  milo_clear_error();

  milo_runtime *rt = milo_runtime_create();
  CHECK(rt != NULL);
  if (rt == NULL) return 1;

  const char *source = "let x = 21\nx * 2\n";
  milo_status status = milo_runtime_check(rt, source, strlen(source), "test.milo");
  CHECK(status == MILO_OK);

  milo_value *value = NULL;
  status = milo_runtime_run(rt, source, strlen(source), "test.milo", &value);
  CHECK(status == MILO_OK);
  if (value != NULL) {
    CHECK(milo_value_kind_of(value) == MILO_VALUE_INT);
    CHECK(milo_value_as_int(value) == 42);
    milo_value_release(value);
  }

  /* A syntax error must be reported, not crash. */
  const char *bad = "let = = =";
  status = milo_runtime_check(rt, bad, strlen(bad), "test.milo");
  CHECK(status == MILO_ERR_SYNTAX);

  char error_buffer[512];
  milo_last_error(error_buffer, sizeof(error_buffer));
  CHECK(strlen(error_buffer) > 0);

  milo_runtime_destroy(rt);
  return failures == 0 ? 0 : 1;
}
