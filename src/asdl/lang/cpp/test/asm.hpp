#if 0
void foo() {
  asm("int3");

  int src=1; int dst ;
  asm("mov %1, %0\n\tadd $1, %0" : "=r"(dst) : "r"(src));


   unsigned int dwRes;
   unsigned int dwSomeValue;

   asm ("bsfl %1,%0" : "=r" (dwRes) : "r" (dwSomeValue) : "cc");

}
#endif


int frob(int x)
{
  int y;
  asm goto ("frob %%r5, %1; jc %l[error]; mov (%2), %%r5"
            : /* No outputs. */
            : "r"(x), "r"(&y)
            : "memory" 
            : error);
  return y;
error:
  return -1;
}
