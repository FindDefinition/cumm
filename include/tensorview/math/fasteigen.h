#pragma once
#include <cmath>
#include <tensorview/tensor.h>

#ifndef SLICE
#define SLICE cslice
#endif


template <typename T>
TV_HOST_DEVICE_INLINE void cross(T* out, const T* a, const T* b){
    out[0] = a[1] * b[2] - a[2] * b[1];
    out[1] = a[2] * b[0] - a[0] * b[2];
    out[2] = a[0] * b[1] - a[1] * b[0];
}

template <typename T>
TV_HOST_DEVICE_INLINE T dot(const T* a, const T* b){
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

template <typename T, int R=3, int C=3>
constexpr T rslice(T i, T j){
    return i * C + j;
}

template <typename T, int R=3, int C=3>
constexpr T cslice(T i, T j){
    return j * R + i;
}

template <typename T>
void TV_HOST_DEVICE ComputeEigenvector0My(const T* A, T eval0, T* out){
    T row0[3], row1[3], row2[3], r0xr1[3], r0xr2[3], r1xr2[3];
    row0[0] = A[SLICE(0, 0)] - eval0;
    row0[1] = A[SLICE(0, 1)];
    row0[2] = A[SLICE(0, 2)];

    row1[0] = A[SLICE(0, 1)];
    row1[1] = A[SLICE(1, 1)] - eval0;
    row1[2] = A[SLICE(1, 2)];

    row2[0] = A[SLICE(0, 2)];
    row2[1] = A[SLICE(1, 2)];
    row2[2] = A[SLICE(2, 2)] - eval0;

    cross(r0xr1, row0, row1);
    cross(r0xr2, row0, row2);
    cross(r1xr2, row1, row2);
    T d0 = dot(r0xr1, r0xr1);
    T d1 = dot(r0xr2, r0xr2);
    T d2 = dot(r1xr2, r1xr2);

    T dmax = d0;
    int imax = 0;
    if (d1 > dmax) {
        dmax = d1;
        imax = 1;
    }
    if (d2 > dmax) {
        imax = 2;
    }
    if (imax == 0) {
        T sqrt_d = sqrt(d0);
        out[0] = r0xr1[0] / sqrt_d;
        out[1] = r0xr1[1] / sqrt_d;
        out[2] = r0xr1[2] / sqrt_d;
    } else if (imax == 1) {
        T sqrt_d = sqrt(d1);
        out[0] = r0xr2[0] / sqrt_d;
        out[1] = r0xr2[1] / sqrt_d;
        out[2] = r0xr2[2] / sqrt_d;
    } else {
        T sqrt_d = sqrt(d2);
        out[0] = r1xr2[0] / sqrt_d;
        out[1] = r1xr2[1] / sqrt_d;
        out[2] = r1xr2[2] / sqrt_d;
    }
}

template <typename T>
void TV_HOST_DEVICE ComputeEigenvector1My(const T* A, const T* evec0, T eval1, T* out){
    T U[3], V[3], AU[3], AV[3];
    if (abs(evec0[0]) > abs(evec0[1])) {
        T inv_length =
                1 / sqrt(evec0[0] * evec0[0] + evec0[2] * evec0[2]);
        U[0] = -evec0[2] * inv_length;
        U[1] = 0;
        U[2] = evec0[0] * inv_length;
    } else {
        T inv_length =
                1 / sqrt(evec0[1] * evec0[1] + evec0[2] * evec0[2]);
        U[0] = 0;
        U[1] = evec0[2] * inv_length;
        U[2] = -evec0[1] * inv_length;
    }
    cross(V, evec0, U);
    AU[0] = A[SLICE(0, 0)] * U[0] + A[SLICE(0, 1)] * U[1] + A[SLICE(0, 2)] * U[2];
    AU[1] = A[SLICE(0, 1)] * U[0] + A[SLICE(1, 1)] * U[1] + A[SLICE(1, 2)] * U[2];
    AU[2] = A[SLICE(0, 2)] * U[0] + A[SLICE(1, 2)] * U[1] + A[SLICE(2, 2)] * U[2];

    AV[0] = A[SLICE(0, 0)] * V[0] + A[SLICE(0, 1)] * V[1] + A[SLICE(0, 2)] * V[2];
    AV[1] = A[SLICE(0, 1)] * V[0] + A[SLICE(1, 1)] * V[1] + A[SLICE(1, 2)] * V[2];
    AV[2] = A[SLICE(0, 2)] * V[0] + A[SLICE(1, 2)] * V[1] + A[SLICE(2, 2)] * V[2];

    T m00 = U[0] * AU[0] + U[1] * AU[1] + U[2] * AU[2] - eval1;
    T m01 = U[0] * AV[0] + U[1] * AV[1] + U[2] * AV[2];
    T m11 = V[0] * AV[0] + V[1] * AV[1] + V[2] * AV[2] - eval1;
    T absM00 = abs(m00);
    T absM01 = abs(m01);
    T absM11 = abs(m11);
    T max_abs_comp;
    if (absM00 >= absM11) {
        max_abs_comp = max(absM00, absM01);
        if (max_abs_comp > 0) {
            if (absM00 >= absM01) {
                m01 /= m00;
                m00 = 1 / sqrt(1 + m01 * m01);
                m01 *= m00;
            } else {
                m00 /= m01;
                m01 = 1 / sqrt(1 + m00 * m00);
                m00 *= m01;
            }
            out[0] = m01 * U[0] - m00 * V[0];
            out[1] = m01 * U[1] - m00 * V[1];
            out[2] = m01 * U[2] - m00 * V[2];
            return;
        } else {
            out[0] = U[0];
            out[1] = U[1];
            out[2] = U[2];
            return;
        }
    } else {
        max_abs_comp = max(absM11, absM01);
        if (max_abs_comp > 0) {
            if (absM11 >= absM01) {
                m01 /= m11;
                m11 = 1 / sqrt(1 + m01 * m01);
                m01 *= m11;
            } else {
                m11 /= m01;
                m01 = 1 / sqrt(1 + m11 * m11);
                m11 *= m01;
            }
            out[0] = m11 * U[0] - m01 * V[0];
            out[1] = m11 * U[1] - m01 * V[1];
            out[2] = m11 * U[2] - m01 * V[2];
        } else {
            out[0] = U[0];
            out[1] = U[1];
            out[2] = U[2];
            return;
        }
    }

}

template <typename T>
void TV_HOST_DEVICE FastEigen3x3MyV2(T* A, T* out){
    T max_coeff = A[0];
    T eval0, eval1, eval2;
    for (int i = 1; i < 9; ++i){
        max_coeff = max(max_coeff, A[i]);
    }
    if (max_coeff == 0){
        out[0] = 0;
        out[1] = 0;
        out[2] = 0;
        return;
    }
    for (int i = 0; i < 9; ++i){
        A[i] /= max_coeff;
    }
    T norm = A[SLICE(0, 1)] * A[SLICE(0, 1)] + A[SLICE(0, 2)] * A[SLICE(0, 2)] + A[SLICE(1, 2)] * A[SLICE(1, 2)];
    if (norm > 0){
        T q = (A[SLICE(0, 0)] + A[SLICE(1, 1)] + A[SLICE(2, 2)]) / 3;

        T b00 = A[SLICE(0, 0)] - q;
        T b11 = A[SLICE(1, 1)] - q;
        T b22 = A[SLICE(2, 2)] - q;

        T p =
                sqrt((b00 * b00 + b11 * b11 + b22 * b22 + norm * 2) / 6);

        T c00 = b11 * b22 - A[SLICE(1, 2)] * A[SLICE(1, 2)];
        T c01 = A[SLICE(0, 1)] * b22 - A[SLICE(1, 2)] * A[SLICE(0, 2)];
        T c02 = A[SLICE(0, 1)] * A[SLICE(1, 2)] - b11 * A[SLICE(0, 2)];
        T det = (b00 * c00 - A[SLICE(0, 1)] * c01 + A[SLICE(0, 2)] * c02) / (p * p * p);

        T half_det = det * 0.5;
        half_det = min(max(half_det, -1.0), 1.0);

        T angle = acosf(half_det) / T(3);
        T const two_thirds_pi = 2.09439510239319549;
        T beta2 = cosf(angle) * 2;
        T beta0 = cosf(angle + two_thirds_pi) * 2;
        T beta1 = -(beta0 + beta2);
        eval0 = q + p * beta0;
        eval1 = q + p * beta1;
        eval2 = q + p * beta2;
        T evec2[3], evec1[3], evec0[3];
        if (half_det >= 0) {
            ComputeEigenvector0My(A, eval2, evec2);
            if (eval2 < eval0 && eval2 < eval1) {
                out[0] = evec2[0];
                out[1] = evec2[1];
                out[2] = evec2[2];
                return;
            }
            ComputeEigenvector1My(A, evec2, eval1, evec1);
            if (eval1 < eval0 && eval1 < eval2) {
                out[0] = evec1[0];
                out[1] = evec1[1];
                out[2] = evec1[2];
                return;
            }
            cross(evec0, evec1, evec2);
            out[0] = evec0[0];
            out[1] = evec0[1];
            out[2] = evec0[2];
            return;
        }else{
            ComputeEigenvector0My(A, eval0, evec0);
            // evec0 = ComputeEigenvector0(A, eval0);
            if (eval0 < eval1 && eval0 < eval2) {
                out[0] = evec0[0];
                out[1] = evec0[1];
                out[2] = evec0[2];
                return;
            }
            ComputeEigenvector1My(A, evec0, eval1, evec1);
            // evec1 = ComputeEigenvector1(A, evec0, eval1);
            if (eval1 < eval0 && eval1 < eval2) {
                out[0] = evec1[0];
                out[1] = evec1[1];
                out[2] = evec1[2];
                return;
            }
            cross(evec2, evec0, evec1);
            out[0] = evec2[0];
            out[1] = evec2[1];
            out[2] = evec2[2];
            return;
        }
    } else {
        for (int i = 0; i < 9; ++i){
            A[i] *= max_coeff;
        }
        if (A[SLICE(0, 0)] < A[SLICE(1, 1)] && A[SLICE(0, 0)] < A[SLICE(2, 2)]) {
            out[0] = 1;
            out[1] = 0;
            out[2] = 0;
            return;
        } else if (A[SLICE(1, 1)] < A[SLICE(0, 0)] && A[SLICE(1, 1)] < A[SLICE(2, 2)]) {
            out[0] = 0;
            out[1] = 1;
            out[2] = 0;
            return;
        } else {
            out[0] = 0;
            out[1] = 0;
            out[2] = 1;
            return;
        }
    }
    
}