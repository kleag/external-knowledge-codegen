static_assert(sizeof(int) > 1, "yes");
#if  __cplusplus >= 201703L
static_assert(sizeof(int));
#endif
