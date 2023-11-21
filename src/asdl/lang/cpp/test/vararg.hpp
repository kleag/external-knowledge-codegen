#include <stdarg.h>

double stddev(int count, ...)
{
    va_list args;
    va_start(args, count);
    for (int i = 0; i < count; ++i) {
        double num = va_arg(args, double);
    }
    va_end(args);
}
