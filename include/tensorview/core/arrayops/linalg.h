#pragma once

#include "simple.h"
#include <tensorview/core/array.h>
#include "mathbase.h"

namespace tv {
namespace arrayops {

template <typename T, size_t N, size_t Align> struct determinant;

template <typename T> struct determinant<array<T, 1, 0>, 1, 0> {
  TV_HOST_DEVICE_INLINE constexpr T
  operator()(const array<array<T, 1>, 1> &self) {
    return self[0][0];
  }
};

template <typename T> struct determinant<array<T, 2, 0>, 2, 0> {
  TV_HOST_DEVICE_INLINE constexpr T
  operator()(const array<array<T, 2>, 2> &self) {
    return self[0][0] * self[1][1] - self[0][1] * self[1][0];
  }
};

template <typename T> struct determinant<array<T, 3, 0>, 3, 0> {
  TV_HOST_DEVICE_INLINE constexpr T operator()(const array<array<T, 3>, 3> &a) {
    return (a[0][0] * (a[1][1] * a[2][2] - a[2][1] * a[1][2]) +
            a[0][1] * (a[1][2] * a[2][0] - a[2][2] * a[1][0]) +
            a[0][2] * (a[1][0] * a[2][1] - a[2][0] * a[1][1]));
  }
};

template <typename T> struct determinant<array<T, 4, 0>, 4, 0> {
  TV_HOST_DEVICE_INLINE constexpr T operator()(const array<array<T, 4>, 4> &a) {
    return a[0][0] *
               (a[1][1] * a[2][2] * a[3][3] + a[3][1] * a[1][2] * a[2][3] +
                a[2][1] * a[3][2] * a[1][3] - a[1][1] * a[3][2] * a[2][3] -
                a[2][1] * a[1][2] * a[3][3] - a[3][1] * a[2][2] * a[1][3]) +
           a[0][1] *
               (a[1][2] * a[3][3] * a[2][0] + a[2][2] * a[1][3] * a[3][0] +
                a[3][2] * a[2][3] * a[1][0] - a[1][2] * a[2][3] * a[3][0] -
                a[3][2] * a[1][3] * a[2][0] - a[2][2] * a[3][3] * a[1][0]) +
           a[0][2] *
               (a[1][3] * a[2][0] * a[3][1] + a[3][3] * a[1][0] * a[2][1] +
                a[2][3] * a[3][0] * a[1][1] - a[1][3] * a[3][0] * a[2][1] -
                a[2][3] * a[1][0] * a[3][1] - a[3][3] * a[2][0] * a[1][1]) +
           a[0][3] *
               (a[1][0] * a[3][1] * a[2][2] + a[2][0] * a[1][1] * a[3][2] +
                a[3][0] * a[2][1] * a[1][2] - a[1][0] * a[2][1] * a[3][2] -
                a[3][0] * a[1][1] * a[2][2] - a[2][0] * a[3][1] * a[1][2]);
  }
};

template <typename T, size_t N, size_t Align> struct adjugate;
template <typename T> struct adjugate<array<T, 1, 0>, 1, 0> {
  TV_HOST_DEVICE_INLINE constexpr array<array<T, 1>, 1>
  operator()(const array<array<T, 1>, 1> &self) {
    return {{T(1)}};
  }
};

template <typename T> struct adjugate<array<T, 2, 0>, 2, 0> {
  TV_HOST_DEVICE_INLINE constexpr array<array<T, 2>, 2>
  operator()(const array<array<T, 2>, 2> &a) {
    return {array<T, 2>{a[1][1], -a[0][1]}, array<T, 2>{-a[1][0], a[0][0]}};
  }
};

template <typename T> struct adjugate<array<T, 3, 0>, 3, 0> {
  TV_HOST_DEVICE_INLINE constexpr array<array<T, 3>, 3>
  operator()(const array<array<T, 3>, 3> &a) {
    return {array<T, 3>{a[1][1] * a[2][2] - a[2][1] * a[1][2],
                        a[2][1] * a[0][2] - a[0][1] * a[2][2],
                        a[0][1] * a[1][2] - a[1][1] * a[0][2]},
            array<T, 3>{a[1][2] * a[2][0] - a[2][2] * a[1][0],
                        a[2][2] * a[0][0] - a[0][2] * a[2][0],
                        a[0][2] * a[1][0] - a[1][2] * a[0][0]},
            array<T, 3>{a[1][0] * a[2][1] - a[2][0] * a[1][1],
                        a[2][0] * a[0][1] - a[0][0] * a[2][1],
                        a[0][0] * a[1][1] - a[1][0] * a[0][1]}};
  }
};

template <typename T> struct adjugate<array<T, 4, 0>, 4, 0> {
  TV_HOST_DEVICE_INLINE constexpr array<array<T, 4>, 4>
  operator()(const array<array<T, 4>, 4> &a) {
    return {array<T, 4>{
                a[1][1] * a[2][2] * a[3][3] + a[3][1] * a[1][2] * a[2][3] +
                    a[2][1] * a[3][2] * a[1][3] - a[1][1] * a[3][2] * a[2][3] -
                    a[2][1] * a[1][2] * a[3][3] - a[3][1] * a[2][2] * a[1][3],
                a[0][1] * a[3][2] * a[2][3] + a[2][1] * a[0][2] * a[3][3] +
                    a[3][1] * a[2][2] * a[0][3] - a[3][1] * a[0][2] * a[2][3] -
                    a[2][1] * a[3][2] * a[0][3] - a[0][1] * a[2][2] * a[3][3],
                a[0][1] * a[1][2] * a[3][3] + a[3][1] * a[0][2] * a[1][3] +
                    a[1][1] * a[3][2] * a[0][3] - a[0][1] * a[3][2] * a[1][3] -
                    a[1][1] * a[0][2] * a[3][3] - a[3][1] * a[1][2] * a[0][3],
                a[0][1] * a[2][2] * a[1][3] + a[1][1] * a[0][2] * a[2][3] +
                    a[2][1] * a[1][2] * a[0][3] - a[0][1] * a[1][2] * a[2][3] -
                    a[2][1] * a[0][2] * a[1][3] - a[1][1] * a[2][2] * a[0][3]},
            array<T, 4>{
                a[1][2] * a[3][3] * a[2][0] + a[2][2] * a[1][3] * a[3][0] +
                    a[3][2] * a[2][3] * a[1][0] - a[1][2] * a[2][3] * a[3][0] -
                    a[3][2] * a[1][3] * a[2][0] - a[2][2] * a[3][3] * a[1][0],
                a[0][2] * a[2][3] * a[3][0] + a[3][2] * a[0][3] * a[2][0] +
                    a[2][2] * a[3][3] * a[0][0] - a[0][2] * a[3][3] * a[2][0] -
                    a[2][2] * a[0][3] * a[3][0] - a[3][2] * a[2][3] * a[0][0],
                a[0][2] * a[3][3] * a[1][0] + a[1][2] * a[0][3] * a[3][0] +
                    a[3][2] * a[1][3] * a[0][0] - a[0][2] * a[1][3] * a[3][0] -
                    a[3][2] * a[0][3] * a[1][0] - a[1][2] * a[3][3] * a[0][0],
                a[0][2] * a[1][3] * a[2][0] + a[2][2] * a[0][3] * a[1][0] +
                    a[1][2] * a[2][3] * a[0][0] - a[0][2] * a[2][3] * a[1][0] -
                    a[1][2] * a[0][3] * a[2][0] - a[2][2] * a[1][3] * a[0][0]},
            array<T, 4>{
                a[1][3] * a[2][0] * a[3][1] + a[3][3] * a[1][0] * a[2][1] +
                    a[2][3] * a[3][0] * a[1][1] - a[1][3] * a[3][0] * a[2][1] -
                    a[2][3] * a[1][0] * a[3][1] - a[3][3] * a[2][0] * a[1][1],
                a[0][3] * a[3][0] * a[2][1] + a[2][3] * a[0][0] * a[3][1] +
                    a[3][3] * a[2][0] * a[0][1] - a[0][3] * a[2][0] * a[3][1] -
                    a[3][3] * a[0][0] * a[2][1] - a[2][3] * a[3][0] * a[0][1],
                a[0][3] * a[1][0] * a[3][1] + a[3][3] * a[0][0] * a[1][1] +
                    a[1][3] * a[3][0] * a[0][1] - a[0][3] * a[3][0] * a[1][1] -
                    a[1][3] * a[0][0] * a[3][1] - a[3][3] * a[1][0] * a[0][1],
                a[0][3] * a[2][0] * a[1][1] + a[1][3] * a[0][0] * a[2][1] +
                    a[2][3] * a[1][0] * a[0][1] - a[0][3] * a[1][0] * a[2][1] -
                    a[2][3] * a[0][0] * a[1][1] - a[1][3] * a[2][0] * a[0][1]},
            array<T, 4>{
                a[1][0] * a[3][1] * a[2][2] + a[2][0] * a[1][1] * a[3][2] +
                    a[3][0] * a[2][1] * a[1][2] - a[1][0] * a[2][1] * a[3][2] -
                    a[3][0] * a[1][1] * a[2][2] - a[2][0] * a[3][1] * a[1][2],
                a[0][0] * a[2][1] * a[3][2] + a[3][0] * a[0][1] * a[2][2] +
                    a[2][0] * a[3][1] * a[0][2] - a[0][0] * a[3][1] * a[2][2] -
                    a[2][0] * a[0][1] * a[3][2] - a[3][0] * a[2][1] * a[0][2],
                a[0][0] * a[3][1] * a[1][2] + a[1][0] * a[0][1] * a[3][2] +
                    a[3][0] * a[1][1] * a[0][2] - a[0][0] * a[1][1] * a[3][2] -
                    a[3][0] * a[0][1] * a[1][2] - a[1][0] * a[3][1] * a[0][2],
                a[0][0] * a[1][1] * a[2][2] + a[2][0] * a[0][1] * a[1][2] +
                    a[1][0] * a[2][1] * a[0][2] - a[0][0] * a[2][1] * a[1][2] -
                    a[1][0] * a[0][1] * a[2][2] - a[2][0] * a[1][1] * a[0][2]}};
  }
};
template <typename T, size_t N, size_t Align> struct row;
template <typename T, size_t N, size_t Align> struct col;

template <typename T, size_t Row, size_t Col>
struct row<array<T, Col>, Row, 0> {
  TV_HOST_DEVICE_INLINE constexpr array<T, Col>
  operator()(const array<array<T, Col>, Row> &self, int idx) const {
    return self[idx];
  }
};

template <typename T, size_t Row, size_t Col>
struct col<array<T, Col>, Row, 0> {
private:
  template <int... Inds>
  TV_HOST_DEVICE_INLINE static constexpr array<T, Row>
  col_impl(const array<array<T, Col>, Row> &a, int idx,
           mp_list_int<Inds...>) noexcept {
    return array<T, Row>{a[Inds][idx]...};
  }

public:
  TV_HOST_DEVICE_INLINE constexpr array<T, Row>
  operator()(const array<array<T, Col>, Row> &self, int idx) {
    return col_impl(self, idx, mp_make_list_c_sequence<int, Row>{});
  }
};

template <typename T, size_t N, size_t Align> struct transpose;

template <typename T, size_t Row, size_t Col>
struct transpose<array<T, Col>, Row, 0> {
private:
  template <int... Inds>
  TV_HOST_DEVICE_INLINE static constexpr array<array<T, Row>, Col>
  transpose_impl(const array<array<T, Col>, Row> &a,
                 mp_list_int<Inds...>) noexcept {
    return array<array<T, Row>, Col>{a.template op<col>(Inds)...};
  }

public:
  TV_HOST_DEVICE_INLINE constexpr array<array<T, Row>, Col>
  operator()(const array<array<T, Col>, Row> &self) {
    return transpose_impl(self, mp_make_list_c_sequence<int, Col>{});
  }
};

template <typename T, size_t N, size_t Align> struct inverse;

template <typename T, size_t N> struct inverse<array<T, N, 0>, N, 0> {
  TV_HOST_DEVICE_INLINE constexpr array<array<T, N>, N>
  operator()(const array<array<T, N>, N> &self) {
    return self.template op<adjugate>() / self.template op<determinant>();
  }
};

template <typename T, size_t N, size_t Align> struct mv_colmajor;

template <typename T, size_t Col> struct mv_colmajor<array<T, Col, 0>, 1, 0> {

  TV_HOST_DEVICE_INLINE constexpr array<T, Col>
  operator()(const array<array<T, Col>, 1> &self, const array<T, 1> &vec) {
    return self.template op<row>(0) * vec[0];
  }
};

template <typename T, size_t Col> struct mv_colmajor<array<T, Col, 0>, 2, 0> {
  TV_HOST_DEVICE_INLINE constexpr array<T, Col>
  operator()(const array<array<T, Col>, 2> &self, const array<T, 2> &vec) {
    return self.template op<row>(0) * vec[0] +
           self.template op<row>(1) * vec[1];
  }
};

template <typename T, size_t Col> struct mv_colmajor<array<T, Col, 0>, 3, 0> {
  TV_HOST_DEVICE_INLINE constexpr array<T, Col>
  operator()(const array<array<T, Col>, 3> &self, const array<T, 3> &vec) {
    return self.template op<row>(0) * vec[0] +
           self.template op<row>(1) * vec[1] +
           self.template op<row>(2) * vec[2];
  }
};

template <typename T, size_t Col> struct mv_colmajor<array<T, Col, 0>, 4, 0> {
  TV_HOST_DEVICE_INLINE constexpr array<T, Col>
  operator()(const array<array<T, Col>, 4> &self, const array<T, 4> &vec) {
    return self.template op<row>(0) * vec[0] +
           self.template op<row>(1) * vec[1] +
           self.template op<row>(2) * vec[2] +
           self.template op<row>(3) * vec[3];
  }
};

template <typename T, size_t N, size_t Align> struct mv_rowmajor;

template <typename T, size_t Row> struct mv_rowmajor<array<T, 1, 0>, Row, 0> {
  TV_HOST_DEVICE_INLINE constexpr array<T, Row>
  operator()(const array<array<T, 1>, Row> &self, const array<T, 1> &vec) {
    return self.template op<col>(0) * vec[0];
  }
};

template <typename T, size_t Row> struct mv_rowmajor<array<T, 2, 0>, Row, 0> {
  TV_HOST_DEVICE_INLINE constexpr array<T, Row>
  operator()(const array<array<T, 2>, Row> &self, const array<T, 2> &vec) {
    return self.template op<col>(0) * vec[0] +
           self.template op<col>(1) * vec[1];
  }
};

template <typename T, size_t Row> struct mv_rowmajor<array<T, 3, 0>, Row, 0> {
  TV_HOST_DEVICE_INLINE constexpr array<T, Row>
  operator()(const array<array<T, 3>, Row> &self, const array<T, 3> &vec) {
    return self.template op<col>(0) * vec[0] +
           self.template op<col>(1) * vec[1] +
           self.template op<col>(2) * vec[2];
  }
};

template <typename T, size_t Row> struct mv_rowmajor<array<T, 4, 0>, Row, 0> {
  TV_HOST_DEVICE_INLINE constexpr array<T, Row>
  operator()(const array<array<T, 4>, Row> &self, const array<T, 4> &vec) {
    return self.template op<col>(0) * vec[0] +
           self.template op<col>(1) * vec[1] +
           self.template op<col>(2) * vec[2] +
           self.template op<col>(3) * vec[3];
  }
};

template <typename T, size_t N, size_t Align> struct mm_nn;

template <typename T, size_t Row, size_t Col>
struct mm_nn<array<T, Row>, Col, 0> {
  // CR @ XC = XR
  TV_HOST_DEVICE_INLINE constexpr array<array<T, Row>, 1>
  operator()(const array<array<T, Row>, Col> &self,
             const array<array<T, Col>, 1> &other) {
    return {self.template op<mv_colmajor>(other.template op<row>(0))};
  }
  TV_HOST_DEVICE_INLINE constexpr array<array<T, Row>, 2>
  operator()(const array<array<T, Row>, Col> &self,
             const array<array<T, Col>, 2> &other) {
    return {self.template op<mv_colmajor>(other.template op<row>(0)),
            self.template op<mv_colmajor>(other.template op<row>(1))};
  }
  TV_HOST_DEVICE_INLINE constexpr array<array<T, Row>, 3>
  operator()(const array<array<T, Row>, Col> &self,
             const array<array<T, Col>, 3> &other) {
    return {self.template op<mv_colmajor>(other.template op<row>(0)),
            self.template op<mv_colmajor>(other.template op<row>(1)),
            self.template op<mv_colmajor>(other.template op<row>(2))};
  }
  TV_HOST_DEVICE_INLINE constexpr array<array<T, Row>, 4>
  operator()(const array<array<T, Row>, Col> &self,
             const array<array<T, Col>, 4> &other) {
    return {self.template op<mv_colmajor>(other.template op<row>(0)),
            self.template op<mv_colmajor>(other.template op<row>(1)),
            self.template op<mv_colmajor>(other.template op<row>(2)),
            self.template op<mv_colmajor>(other.template op<row>(3))};
  }
};

template <typename T, size_t N, size_t Align> struct mm_tt_v1;

template <typename T, size_t Col, size_t RowOther>
struct mm_tt_v1<array<T, Col>, RowOther, 0> {
  // XC @ CR = XR
  TV_HOST_DEVICE_INLINE constexpr array<array<T, 1>, RowOther>
  operator()(const array<array<T, Col>, RowOther> &self,
             const array<array<T, 1>, Col> &other) {
    return other.template op<mm_nn>(self);
  }
  TV_HOST_DEVICE_INLINE constexpr array<array<T, 2>, RowOther>
  operator()(const array<array<T, Col>, RowOther> &self,
             const array<array<T, 2>, Col> &other) {
    return other.template op<mm_nn>(self);
  }
  TV_HOST_DEVICE_INLINE constexpr array<array<T, 3>, RowOther>
  operator()(const array<array<T, Col>, RowOther> &self,
             const array<array<T, 3>, Col> &other) {
    return other.template op<mm_nn>(self);
  }
  TV_HOST_DEVICE_INLINE constexpr array<array<T, 4>, RowOther>
  operator()(const array<array<T, Col>, RowOther> &self,
             const array<array<T, 4>, Col> &other) {
    return other.template op<mm_nn>(self);
  }
};

template <typename T, size_t N, size_t Align> struct mm_tt;

template <typename T, size_t Col, size_t RowOther>
struct mm_tt<array<T, Col>, RowOther, 0> {
  // XC @ CR = XR
  TV_HOST_DEVICE_INLINE constexpr array<array<T, 1>, RowOther>
  operator()(const array<array<T, Col>, RowOther> &self,
             const array<array<T, 1>, Col> &other) {
    return array<array<T, RowOther>, 1>{
        self.template op<mv_rowmajor>(other.template op<col>(0))}
        .template op<transpose>();
  }
  TV_HOST_DEVICE_INLINE constexpr array<array<T, 2>, RowOther>
  operator()(const array<array<T, Col>, RowOther> &self,
             const array<array<T, 2>, Col> &other) {
    return array<array<T, RowOther>, 2>{
        self.template op<mv_rowmajor>(other.template op<col>(0)),
        self.template op<mv_rowmajor>(other.template op<col>(1))}
        .template op<transpose>();
  }
  TV_HOST_DEVICE_INLINE constexpr array<array<T, 3>, RowOther>
  operator()(const array<array<T, Col>, RowOther> &self,
             const array<array<T, 3>, Col> &other) {
    return array<array<T, RowOther>, 3>{
        self.template op<mv_rowmajor>(other.template op<col>(0)),
        self.template op<mv_rowmajor>(other.template op<col>(1)),
        self.template op<mv_rowmajor>(other.template op<col>(2))}
        .template op<transpose>();
  }
  TV_HOST_DEVICE_INLINE constexpr array<array<T, 4>, RowOther>
  operator()(const array<array<T, Col>, RowOther> &self,
             const array<array<T, 4>, Col> &other) {
    return array<array<T, RowOther>, 4>{
        self.template op<mv_rowmajor>(other.template op<col>(0)),
        self.template op<mv_rowmajor>(other.template op<col>(1)),
        self.template op<mv_rowmajor>(other.template op<col>(2)),
        self.template op<mv_rowmajor>(other.template op<col>(3))}
        .template op<transpose>();
  }
};

template <typename T, size_t N, size_t Align> struct mm_tn;

template <typename T, size_t Col, size_t RowOther>
struct mm_tn<array<T, Col>, RowOther, 0> {
  // XC @ CR = XR
  TV_HOST_DEVICE_INLINE constexpr array<array<T, 1>, RowOther>
  operator()(const array<array<T, Col>, RowOther> &self,
             const array<array<T, Col>, 1> &other) {
    return array<array<T, RowOther>, 1>{
        self.template op<mv_rowmajor>(other.template op<row>(0))}
        .template op<transpose>();
  }
  TV_HOST_DEVICE_INLINE constexpr array<array<T, 2>, RowOther>
  operator()(const array<array<T, Col>, RowOther> &self,
             const array<array<T, Col>, 2> &other) {
    return array<array<T, RowOther>, 2>{
        self.template op<mv_rowmajor>(other.template op<row>(0)),
        self.template op<mv_rowmajor>(other.template op<row>(1))}
        .template op<transpose>();
  }
  TV_HOST_DEVICE_INLINE constexpr array<array<T, 3>, RowOther>
  operator()(const array<array<T, Col>, RowOther> &self,
             const array<array<T, Col>, 3> &other) {
    return array<array<T, RowOther>, 3>{
        self.template op<mv_rowmajor>(other.template op<row>(0)),
        self.template op<mv_rowmajor>(other.template op<row>(1)),
        self.template op<mv_rowmajor>(other.template op<row>(2))}
        .template op<transpose>();
  }
  TV_HOST_DEVICE_INLINE constexpr array<array<T, 4>, RowOther>
  operator()(const array<array<T, Col>, RowOther> &self,
             const array<array<T, Col>, 4> &other) {
    return array<array<T, RowOther>, 4>{
        self.template op<mv_rowmajor>(other.template op<row>(0)),
        self.template op<mv_rowmajor>(other.template op<row>(1)),
        self.template op<mv_rowmajor>(other.template op<row>(2)),
        self.template op<mv_rowmajor>(other.template op<row>(3))}
        .template op<transpose>();
  }
};

namespace arrayops_detail {
template <typename T, size_t Row, size_t Col> struct transform_3d_impl {
  template <int... Inds>
  TV_HOST_DEVICE_INLINE static constexpr array<array<T, Col>, Row>
  run(const array<array<T, Col>, Row> &a, const array<array<T, 3>, 3> &b,
      mp_list_int<Inds...>) noexcept {
    return array<array<T, Col>, Row>{
        (concat(b.template op<mv_rowmajor>(slice<0, 3>(a[Inds])),
                slice<3, Col>(a[Inds])))...};
  }
};

template <typename T, size_t Row> struct transform_3d_impl<T, Row, 3> {
  template <int... Inds>
  TV_HOST_DEVICE_INLINE static constexpr array<array<T, 3>, Row>
  run(const array<array<T, 3>, Row> &a, const array<array<T, 3>, 3> &b,
      mp_list_int<Inds...>) noexcept {
    return array<array<T, 3>, Row>{(b.template op<mv_rowmajor>(a[Inds]))...};
  }
};

template <typename T, size_t Row, size_t Col> struct add_offset_impl {
  template <int... Inds>
  TV_HOST_DEVICE_INLINE static constexpr array<array<T, Col>, Row>
  run(const array<array<T, Col>, Row> &a, const array<T, 3> &b,
      mp_list_int<Inds...>) noexcept {
    return array<array<T, Col>, Row>{
        concat((slice<0, 3>(a[Inds]) + b), slice<3, Col>(a[Inds]))...};
  }
};

template <typename T, size_t Row> struct add_offset_impl<T, Row, 3> {
  template <int... Inds>
  TV_HOST_DEVICE_INLINE static constexpr array<array<T, 3>, Row>
  run(const array<array<T, 3>, Row> &a, const array<T, 3> &b,
      mp_list_int<Inds...>) noexcept {
    return array<array<T, 3>, Row>{(a[Inds] + b)...};
  }
};

} // namespace arrayops_detail

template <typename T, size_t N, size_t Align> struct add_offset_3d;

template <typename T, size_t Row, size_t Col>
struct add_offset_3d<array<T, Col>, Row, 0> {
public:
  TV_HOST_DEVICE_INLINE constexpr array<array<T, Col>, Row>
  operator()(const array<array<T, Col>, Row> &self, const array<T, 3> &other) {
    return arrayops_detail::add_offset_impl<T, Row, Col>::run(
        self, other, mp_make_list_c_sequence<int, Row>{});
  }
};

template <typename T, size_t Col> struct add_offset_3d<T, Col, 0> {
  TV_HOST_DEVICE_INLINE constexpr array<T, Col>
  operator()(const array<T, Col> &self, const array<T, 3> &other) {
    return concat((slice<0, 3>(self) + other), slice<3, Col>(self));
  }
};

template <typename T> struct add_offset_3d<T, 3, 0> {
  TV_HOST_DEVICE_INLINE constexpr array<T, 3>
  operator()(const array<T, 3> &self, const array<T, 3> &other) {
    return self + other;
  }
};

template <typename T, size_t N, size_t Align> struct transform_3d;

template <typename T, size_t Row, size_t Col>
struct transform_3d<array<T, Col>, Row, 0> {
public:
  TV_HOST_DEVICE_INLINE constexpr array<array<T, Col>, Row>
  operator()(const array<array<T, Col>, Row> &self,
             const array<array<T, 3>, 3> &other) {
    return arrayops_detail::transform_3d_impl<T, Row, Col>::run(
        self, other, mp_make_list_c_sequence<int, Row>{});
  }
  TV_HOST_DEVICE_INLINE constexpr array<array<T, Col>, Row>
  operator()(const array<array<T, Col>, Row> &self,
             const array<array<T, 4>, 4> &other) {
    return arrayops_detail::transform_3d_impl<T, Row, Col>::run(
               self, slice_2d<0, 0, 3, 3>(other),
               mp_make_list_c_sequence<int, Row>{})
        .template op<add_offset_3d>(slice<0, 3>(other.template op<col>(3)));
  }
};

template <typename T, size_t Col> struct transform_3d<T, Col, 0> {
public:
  TV_HOST_DEVICE_INLINE constexpr array<T, Col>
  operator()(const array<T, Col> &self, const array<array<T, 3>, 3> &other) {
    return concat(other.template op<mv_rowmajor>(slice<0, 3>(self)),
                  slice<3, Col>(self));
  }

  TV_HOST_DEVICE_INLINE constexpr array<T, Col>
  operator()(const array<T, Col> &self, const array<array<T, 4>, 4> &other) {
    return operator()(self, {slice<0, 3>(other[0]), slice<0, 3>(other[1]),
                             slice<0, 3>(other[2])})
        .template op<add_offset_3d>(slice<0, 3>(other.template op<col>(3)));
  }
};

template <typename T> struct transform_3d<T, 3, 0> {
public:
  TV_HOST_DEVICE_INLINE constexpr array<T, 3>
  operator()(const array<T, 3> &self, const array<array<T, 3>, 3> &other) {
    return other.template op<mv_rowmajor>(self);
  }

  TV_HOST_DEVICE_INLINE constexpr array<T, 3>
  operator()(const array<T, 3> &self, const array<array<T, 4>, 4> &other) {
    return operator()(self, {slice<0, 3>(other[0]), slice<0, 3>(other[1]),
                             slice<0, 3>(other[2])})
        .template op<add_offset_3d>(slice<0, 3>(other.template op<col>(3)));
  }
};

template <typename T, size_t N, size_t Align> struct qxdir;
template <typename T, size_t N, size_t Align> struct qydir;
template <typename T, size_t N, size_t Align> struct qzdir;
template <typename T, size_t N, size_t Align> struct qangle;
template <typename T, size_t N, size_t Align> struct qaxis;

template <typename T> struct qxdir<T, 4, 0> {
  TV_HOST_DEVICE_INLINE constexpr array<T, 3> operator()(const array<T, 4> &q) {
    return {q[3] * q[3] + q[0] * q[0] - q[1] * q[1] - q[2] * q[2],
            (q[0] * q[1] + q[2] * q[3]) * 2, (q[2] * q[0] - q[1] * q[3]) * 2};
  }
};

template <typename T> struct qydir<T, 4, 0> {
  TV_HOST_DEVICE_INLINE constexpr array<T, 3> operator()(const array<T, 4> &q) {
    return {(q[0] * q[1] - q[2] * q[3]) * 2,
            q[3] * q[3] - q[0] * q[0] + q[1] * q[1] - q[2] * q[2],
            (q[1] * q[2] + q[0] * q[3]) * 2};
  }
};

template <typename T> struct qaxis<T, 4, 0> {
  TV_HOST_DEVICE_INLINE constexpr array<T, 3> operator()(const array<T, 4> &q) {
    return slice<0, 3>(q).template op<normalize>();
  }
};

template <typename T> struct qangle<T, 4, 0> {
  TV_HOST_DEVICE_INLINE constexpr T operator()(const array<T, 4> &q) {

    return MathScalarOp<T>::atan2(slice<0, 3>(q).template op<l2norm>(), q[3]) * T(2);
  }
};

template <typename T> struct qzdir<T, 4, 0> {
  TV_HOST_DEVICE_INLINE constexpr array<T, 3> operator()(const array<T, 4> &q) {
    return {(q[2] * q[0] + q[1] * q[3]) * 2, (q[1] * q[2] - q[0] * q[3]) * 2,
            q[3] * q[3] - q[0] * q[0] - q[1] * q[1] + q[2] * q[2]};
  }
};
template <typename T, size_t N, size_t Align> struct rotation_quat;

template <typename T> struct rotation_quat<T, 3, 0> {
  TV_HOST_DEVICE_INLINE array<T, 4>
  operator()(const array<T, 3> &self, T angle) {
    return concat(self * MathScalarOp<T>::sin(angle / 2),
                  create_array(MathScalarOp<T>::cos(angle / 2)));
  }
};

// template <typename T, size_t N, size_t Align> struct rotation_quat_matrix_simple;

// template <typename T> struct rotation_quat_matrix_simple<array<T, 3>, 3, 0> {
//   TV_HOST_DEVICE_INLINE array<T, 4>
//   operator()(const array<array<T, 3>, 3> &m) {
//     array<T, 4> q{m[0][0]-m[1][1]-m[2][2], m[1][1]-m[0][0]-m[2][2], m[2][2]-m[0][0]-m[1][1], m[0][0]+m[1][1]+m[2][2]};
//     array<array<T, 4>, 4> s{
//         array<T, 4>{T(1), m[0][1] + m[1][0], m[2][0] + m[0][2], m[2][1] - m[1][2]}, 
//         array<T, 4>{m[0][1] + m[1][0], T(1), m[1][2] + m[2][1], m[0][2] - m[2][0]},
//         array<T, 4>{m[0][2] + m[2][0], m[1][2] + m[2][1], T(1), m[1][0] - m[0][1]},
//         array<T, 4>{m[2][1] - m[1][2], m[0][2] - m[2][0], m[1][0] - m[0][1], T(1)}
//     };

//     int j = 0;
//     for (int i = 0; i < 4; ++i){
//       if(q[i] > q[j]) 
//         j = i;
//     }
//     return apply(MathScalarOp<T>::copysign, apply(MathScalarOp<T>::sqrt, (q + T(1)).template op<arrayops::max>(T(0))).template op<normalize>(), s[j]);
//   }
// };


template <typename T> struct rotation_quat<array<T, 3>, 3, 0> {
  TV_HOST_DEVICE_INLINE array<T, 4>
  operator()(const array<array<T, 3>, 3> &m) {
    T t = m[0][0] + m[1][1] + m[2][2];
    array<T, 4> q;
    if (t > T(0))
    {
      t = MathScalarOp<T>::sqrt(t + T(1.0));
      q[3] = T(0.5)*t;
      t = T(0.5)/t;
      q[0] = (m[2][1] - m[1][2]) * t;
      q[1] = (m[0][2] - m[2][0]) * t;
      q[2] = (m[1][0] - m[0][1]) * t;
    }
    else
    {
      int i = 0;
      if (m[1][1] > m[0][0])
        i = 1;
      if (m[2][2] > m[i][i])
        i = 2;
      int j = (i+1)%3;
      int k = (j+1)%3;

      t = MathScalarOp<T>::sqrt(m[i][i]-m[j][j]-m[k][k] + T(1.0));
      q[i] = T(0.5) * t;
      q[i] = T(0.5) * t;
      t = T(0.5)/t;
      q[3] = (m[k][j]-m[j][k])*t;
      q[j] = (m[j][i]+m[i][j])*t;
      q[k] = (m[k][i]+m[i][k])*t;
    }

    return q;
  }
};

template <typename T, size_t N, size_t Align> struct qmat;

template <typename T> struct qmat<T, 4, 0> {
  TV_HOST_DEVICE_INLINE constexpr array<array<T, 3>, 3>
  operator()(const array<T, 4> &self) {
    return (array<array<T, 3>, 3>{self.template op<qxdir>(),
            self.template op<qydir>(),
            self.template op<qzdir>()}).template op<transpose>();
  }
};

template <typename T, size_t N, size_t Align> struct angleaxis_mat;

template <typename T> struct angleaxis_mat<T, 3, 0> {
  TV_HOST_DEVICE_INLINE array<array<T, 3>, 3>
  operator()(const array<T, 3> &m_axis, T m_angle) {
    array<array<T, 3>, 3> res;
    array<T, 3> sin_axis = MathScalarOp<T>::sin(m_angle) * m_axis;
    T c = MathScalarOp<T>::cos(m_angle);
    array<T, 3> cos1_axis = (T(1)-c) * m_axis;

    T tmp;
    tmp = cos1_axis[0] * m_axis[1];
    res[0][1] = tmp - sin_axis[2];
    res[1][0] = tmp + sin_axis[2];

    tmp = cos1_axis[0] * m_axis[2];
    res[0][2] = tmp + sin_axis[1];
    res[2][0] = tmp - sin_axis[1];

    tmp = cos1_axis[1] * m_axis[2];
    res[1][2] = tmp - sin_axis[0];
    res[2][1] = tmp + sin_axis[0];
    auto tmp2 = cos1_axis * m_axis;
    res[0][0] = tmp2[0] + c;
    res[1][1] = tmp2[1] + c;
    res[2][2] = tmp2[2] + c;
    return res;
  }
};


// constexpr tv::array_nd<float, 3, 3> a{tv::array<float, 3>{1, 2, 3},
//                                               tv::array<float, 3>{4, 5, 6},
//                                               tv::array<float, 3>{7, 8, 3}};
// constexpr tv::array_nd<float, 4, 3> a2{tv::array<float, 3>{1, 2, 3},
//                                               tv::array<float, 3>{4, 5, 6},
//                                               tv::array<float, 3>{7, 8, 3},
//                                               tv::array<float, 3>{3, 5, 2}};
// constexpr tv::array_nd<float, 8, 3> a3{};
// constexpr auto a_slice = slice_2d<0, 0, 2, 2>(a);
// constexpr tv::array_nd<float, 3, 3> b{tv::array<float, 3>{9, 7, 8},
//                                               tv::array<float, 3>{6, 5, 4},
//                                               tv::array<float, 3>{3, 2, 1}};
// constexpr tv::array_nd<float, 4, 4> b_4x4{tv::array<float, 4>{9, 7, 8, 0},
//                                               tv::array<float, 4>{6, 5, 4,
//                                               0}, tv::array<float, 4>{3, 2,
//                                               1, 0}, tv::array<float, 4>{0,
//                                               0, 0, 1}};

// constexpr auto c = a.op<mm_tt>(b);
// constexpr int rank = detail::get_tv_array_rank<decltype(a)>::value;
// // constexpr auto inv_a = a.op<row>(0);
// constexpr auto x = a.op<determinant>();
// constexpr auto y = a.op<inverse>();

// constexpr tv::array<tv::array<float, 3UL>, 3UL> inv_a2 = a.op<adjugate>();
// constexpr auto rtx =
//     tv::arrayops::apply(tv::detail::array_div<float>, inv_a2, x);

// constexpr auto rtx2 = inv_a2 / x;

// constexpr auto cc = a.op<transform_3d>(b_4x4);
// constexpr auto cccc = a3.op<transform_3d>(b_4x4);

// constexpr auto ccc = a[0].op<transform_3d>(b);

} // namespace arrayops
} // namespace tv