#ifndef vmath_print_h
#define vmath_print_h


#define NEWLINE fputc ('\n', stderr)
#define RED "\x1b[33m"
#define GREEN "\x1b[32m"
#define YELLOW "\x1b[34m"
#define CLEAR "\x1b[0m"


#define print_vector(n, vct) do {                                   \
    NEWLINE;                                                        \
    fprintf (stderr, "%6c", ' ');                                   \
    for (size_t i = 0; i < n; i++) {                                \
        fprintf (stderr, YELLOW "%6c[%2zu] " CLEAR, ' ', i);        \
    }                                                               \
    NEWLINE;                                                        \
    fprintf (stderr, "%6c", ' ');                                   \
    for (size_t i = 0; i < n; i++) {                                \
        fprintf (stderr, "%10.5Lf ",  (scalar)(vct)[i]);            \
    }                                                               \
    NEWLINE;                                                        \
} while (0)


#define print_vector_exp(n, vct) do {                               \
    NEWLINE;                                                        \
    fprintf (stderr, "%6c", ' ');                                   \
    for (size_t i = 0; i < n; i++) {                                \
        fprintf (stderr, YELLOW "%6c[%2zu] " CLEAR, ' ', i);        \
    }                                                               \
    NEWLINE;                                                        \
    fprintf (stderr, "%6c", ' ');                                   \
    for (size_t i = 0; i < n; i++) {                                \
        scalar val = expl ((scalar)(vct)[i]);                       \
        fprintf (stderr, "%10.5Lf ",  val);                         \
    }                                                               \
    NEWLINE;                                                        \
} while (0)


#define print_matrix(rows, cols, mtx) do {                          \
    NEWLINE;                                                        \
    fprintf (stderr, "%6c", ' ');                                   \
    for (size_t i = 0; i < cols; i++) {                             \
        fprintf (stderr, GREEN "%6c[%2zu] " CLEAR, ' ', i);         \
    }                                                               \
    NEWLINE;                                                        \
    for (size_t i = 0; i < rows; i++) {                             \
        fprintf (stderr, GREEN "[%3zu] " CLEAR, i);                 \
        for (size_t j = 0; j < cols; j++) {                         \
            fprintf (stderr, "%10.5Lf ", (scalar)(mtx)[i*cols+j]);  \
        }                                                           \
        NEWLINE;                                                    \
    }                                                               \
} while (0)


#define print_matrix_exp(rows, cols, mtx) do {                      \
    NEWLINE;                                                        \
    fprintf (stderr, "%6c", ' ');                                   \
    for (size_t i = 0; i < cols; i++) {                             \
        fprintf (stderr, GREEN "%6c[%2zu] " CLEAR, ' ', i);         \
    }                                                               \
    NEWLINE;                                                        \
    for (size_t i = 0; i < rows; i++) {                             \
        fprintf (stderr, GREEN "[%3zu] " CLEAR, i);                 \
        for (size_t j = 0; j < cols; j++) {                         \
            scalar val = expl ((scalar)(mtx)[i*cols+j]);            \
            fprintf (stderr, "%10.5Lf ", val);                      \
        }                                                           \
        NEWLINE;                                                    \
    }                                                               \
} while (0)


#endif  /* vmath_print_h */
