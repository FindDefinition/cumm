#pragma once

#include <tensorview/core/array.h>

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