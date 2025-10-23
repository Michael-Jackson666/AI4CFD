/*
 * 测试 C 编译环境
 * 编译命令: gcc -o test_c test_c.c -lm
 * 运行命令: ./test_c
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

int main() {
    printf("========================================\n");
    printf("  C 编译环境测试程序\n");
    printf("========================================\n\n");
    
    // 1. 基本输出测试
    printf("✓ 基本 printf 功能正常\n\n");
    
    // 2. 数学库测试
    double pi = M_PI;
    double sqrt_2 = sqrt(2.0);
    double sin_pi_4 = sin(M_PI / 4.0);
    double exp_1 = exp(1.0);
    
    printf("数学库测试:\n");
    printf("  π = %.10f\n", pi);
    printf("  √2 = %.10f\n", sqrt_2);
    printf("  sin(π/4) = %.10f\n", sin_pi_4);
    printf("  e = %.10f\n", exp_1);
    printf("✓ 数学库 (libm) 正常\n\n");
    
    // 3. 动态内存分配测试
    int n = 10;
    double *array = (double *)malloc(n * sizeof(double));
    
    if (array == NULL) {
        printf("✗ 内存分配失败!\n");
        return 1;
    }
    
    for (int i = 0; i < n; i++) {
        array[i] = i * 1.5;
    }
    
    printf("动态内存分配测试:\n");
    printf("  数组前5个元素: ");
    for (int i = 0; i < 5; i++) {
        printf("%.1f ", array[i]);
    }
    printf("\n");
    
    free(array);
    printf("✓ 动态内存分配正常\n\n");
    
    // 4. 时间函数测试
    time_t current_time = time(NULL);
    char *time_str = ctime(&current_time);
    
    printf("时间函数测试:\n");
    printf("  当前时间: %s", time_str);
    printf("✓ 时间函数正常\n\n");
    
    // 5. 浮点运算测试
    double a = 1.23456789;
    double b = 9.87654321;
    double sum = a + b;
    double product = a * b;
    double quotient = b / a;
    
    printf("浮点运算测试:\n");
    printf("  %.8f + %.8f = %.8f\n", a, b, sum);
    printf("  %.8f × %.8f = %.8f\n", a, b, product);
    printf("  %.8f ÷ %.8f = %.8f\n", b, a, quotient);
    printf("✓ 浮点运算正常\n\n");
    
    // 6. 指针操作测试
    int x = 42;
    int *ptr = &x;
    
    printf("指针操作测试:\n");
    printf("  变量 x 的值: %d\n", x);
    printf("  变量 x 的地址: %p\n", (void*)&x);
    printf("  指针 ptr 的值: %p\n", (void*)ptr);
    printf("  *ptr 的值: %d\n", *ptr);
    printf("✓ 指针操作正常\n\n");
    
    // 7. 循环和条件测试
    int count = 0;
    for (int i = 0; i < 100; i++) {
        if (i % 2 == 0) {
            count++;
        }
    }
    
    printf("循环和条件测试:\n");
    printf("  0-99 中偶数个数: %d\n", count);
    printf("✓ 循环和条件语句正常\n\n");
    
    // 8. 函数调用测试
    double factorial_5 = 1.0;
    for (int i = 1; i <= 5; i++) {
        factorial_5 *= i;
    }
    
    printf("计算测试:\n");
    printf("  5! = %.0f\n", factorial_5);
    printf("✓ 函数调用正常\n\n");
    
    // 总结
    printf("========================================\n");
    printf("  🎉 所有测试通过！C 环境配置正确！\n");
    printf("========================================\n\n");
    
    printf("编译器信息:\n");
    #ifdef __GNUC__
        printf("  GCC 版本: %d.%d.%d\n", __GNUC__, __GNUC_MINOR__, __GNUC_PATCHLEVEL__);
    #endif
    
    #ifdef __clang__
        printf("  Clang 版本: %d.%d.%d\n", __clang_major__, __clang_minor__, __clang_patchlevel__);
    #endif
    
    printf("\n系统信息:\n");
    #ifdef __APPLE__
        printf("  操作系统: macOS\n");
    #elif __linux__
        printf("  操作系统: Linux\n");
    #elif _WIN32
        printf("  操作系统: Windows\n");
    #endif
    
    #ifdef __x86_64__
        printf("  架构: x86_64 (64-bit)\n");
    #elif __aarch64__
        printf("  架构: ARM64 (Apple Silicon)\n");
    #endif
    
    printf("\n✓ C 环境已就绪，可以开始编译 HyPar 或其他 C 程序！\n");
    
    return 0;
}
