#pragma once

template <typename T> void random_matrix(T *mat, int n) {
    for (int i = 0; i < n * n; i++) {
        mat[i] = static_cast<T>(rand()) / RAND_MAX * 20.0 - 10.0;
    }
}

template <typename T> T random_value() { return static_cast<T>(rand()) / RAND_MAX * 20.0 - 10.0; }