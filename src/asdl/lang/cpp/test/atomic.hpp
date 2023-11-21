void foo(int * ptr) {
  int val;
  int ret;
  int expected;
  int desired;
  bool test;

  __atomic_load_n(ptr, __ATOMIC_SEQ_CST);
  __atomic_load(ptr, &ret, __ATOMIC_RELAXED);
  __atomic_store_n(ptr, 1, __ATOMIC_SEQ_CST);
  __atomic_store(ptr, &val, __ATOMIC_SEQ_CST);
  __atomic_exchange_n (ptr, val, __ATOMIC_CONSUME);
  __atomic_exchange (ptr, &val, &ret, __ATOMIC_ACQUIRE);
  __atomic_compare_exchange_n(ptr, &expected, desired, true, __ATOMIC_SEQ_CST, __ATOMIC_RELAXED);
  __atomic_compare_exchange (ptr, &expected, &desired, false, __ATOMIC_SEQ_CST, __ATOMIC_RELAXED);
  __atomic_add_fetch(ptr, val, __ATOMIC_SEQ_CST);
  __atomic_sub_fetch(ptr, val, __ATOMIC_SEQ_CST);
  __atomic_and_fetch(ptr, val, __ATOMIC_SEQ_CST);
  __atomic_xor_fetch(ptr, val, __ATOMIC_SEQ_CST);
  __atomic_or_fetch(ptr, val, __ATOMIC_SEQ_CST);
  __atomic_nand_fetch(ptr, val, __ATOMIC_SEQ_CST);
  __atomic_fetch_add(ptr, val, __ATOMIC_SEQ_CST);
  __atomic_fetch_sub(ptr, val, __ATOMIC_SEQ_CST);
  __atomic_fetch_and(ptr, val, __ATOMIC_SEQ_CST);
  __atomic_fetch_xor(ptr, val, __ATOMIC_SEQ_CST);
  __atomic_fetch_or(ptr, val, __ATOMIC_SEQ_CST);
  __atomic_fetch_nand(ptr, val, __ATOMIC_SEQ_CST);
  __atomic_test_and_set(ptr, __ATOMIC_SEQ_CST);
  __atomic_clear(&test, __ATOMIC_SEQ_CST);
  __atomic_thread_fence(__ATOMIC_SEQ_CST);
  __atomic_signal_fence(__ATOMIC_SEQ_CST);
  __atomic_always_lock_free(4, ptr);
  __atomic_is_lock_free(4, ptr);
}

