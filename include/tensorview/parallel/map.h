#pragma once 

#include "kernel1d.h"

#include <tensorview/tensor.h>

namespace tv {

namespace detail {

template <typename T, int Rank>
struct CastTensorViewToAccessor {
    static TV_HOST_DEVICE_INLINE decltype(auto) run(tv::TensorView<T, Rank> tview, size_t i){
        return tview.accessor(i);
    }
};

template <typename T, int Rank>
struct CastTensorToTview {
    static TV_HOST_DEVICE_INLINE tv::TensorView<T, Rank> run(tv::Tensor ten){
        return ten.tview<T, Rank>();
    }
};

template <class Types, int... Ranks, typename F, class... Tensors>
void map_first_axis(F&& f, Tensors&&... tens){
    auto tensors = std::forward_as_tuple(tens...);
    constexpr auto ranks = array<int, sizeof...(Ranks)>{Ranks...};
    static_assert(sizeof...(Ranks) == sizeof...(Tensors), "errror");
    auto& first_tensor = std::get<0>(tensors);
    TV_ASSERT_INVALID_ARG(first_tensor.device() == -1, "only support cpu tensor for now.");
    mp_for_each<mp_list_int_range<0, sizeof...(Ranks)>>([&](auto IV){
        constexpr int kIndex = TV_DECLTYPE(IV)::value;
        auto& ten = std::get<kIndex>(tensors);
        TV_ASSERT_INVALID_ARG(ten.device() == first_tensor.device(), "all tensor must have same device");
        TV_ASSERT_INVALID_ARG(ten.dtype() == first_tensor.dtype(), "all tensor must have same dtype");
        TV_ASSERT_INVALID_ARG(ten.ndim() == ranks[kIndex], "rank must match");

    });
    tv::Dispatch<Types>()(first_tensor.dtype(), [&](auto IV){
        using T = TV_DECLTYPE(IV);
        // cuda don't support generic extend lambda.
        tv::kernel_1d_map_cpu(first_tensor.device(), first_tensor.dim(0), [=](size_t i, auto... args){
            f(i, CastTensorViewToAccessor<T, Ranks>::run(args, i)...);
        }, CastTensorToTview<T, Ranks>::run(tens)...);
    });
}

}

template <int... Ranks, typename F, class... Tensors>
void map_first_axis_floats(F&& f, Tensors&&... tens){
    return detail::map_first_axis<tv::mp_list<float, double>, Ranks...>(std::forward<F>(f), std::forward<Tensors>(tens)...);
}


}