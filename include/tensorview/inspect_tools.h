// Copyright 2021 Yan Yan
// 
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// 
//     http://www.apache.org/licenses/LICENSE-2.0
// 
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "cc17.h"
#include "mp_helper.h"
#include <array>
#include <iostream>
#include <map>
#include <set>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <vector>
namespace tv {
namespace detail {
template <typename Os, typename T, typename = void>
struct is_ostream_available : std::false_type {};

template <typename Os, typename T>
struct is_ostream_available<
    Os, T,
    void_t<decltype(operator<<(std::declval<Os &>(),
                               std::declval<const T &>()))>> : std::true_type {
};

template <typename T> struct ObjectPrinter {
  template <typename Os> static Os &run(Os &os, const T &obj) {
    if_constexpr<is_ostream_available<Os, T>::value>(
        [&](auto _) { _(os) << _(obj); }, [&](auto _) { os << "Unkwn"; });
    return os;
  }
};
z
#define DECLARE_CPP_DEFAULT_TYPES(T)                                           \
  template <> struct ObjectPrinter<T> {                                        \
    template <typename Os> static inline Os &run(Os &os, const T &obj) {       \
      return os << obj;                                                        \
    }                                                                          \
  }

    DECLARE_CPP_DEFAULT_TYPES(int8_t);
DECLARE_CPP_DEFAULT_TYPES(int16_t);
DECLARE_CPP_DEFAULT_TYPES(int32_t);
DECLARE_CPP_DEFAULT_TYPES(int64_t);
DECLARE_CPP_DEFAULT_TYPES(uint8_t);
DECLARE_CPP_DEFAULT_TYPES(uint32_t);
DECLARE_CPP_DEFAULT_TYPES(uint64_t);
DECLARE_CPP_DEFAULT_TYPES(uint16_t);
DECLARE_CPP_DEFAULT_TYPES(float);
DECLARE_CPP_DEFAULT_TYPES(double);
DECLARE_CPP_DEFAULT_TYPES(std::string);

#undef DECLARE_CPP_DEFAULT_TYPES

template <> struct ObjectPrinter<bool> {
  template <typename Os> static inline Os &run(Os &os, const bool &obj) {
    return os << (obj ? "True" : "False");
  }
};

template <typename T, std::size_t N> struct ObjectPrinter<std::array<T, N>> {
  template <typename Os> static Os &run(Os &os, const std::array<T, N> &obj) {
    os << "[";
    for (auto it = obj.cbegin(); it != obj.cend(); ++it) {
      ObjectPrinter<T>::run(os, *it);
      if (std::next(it) != obj.cend()) {
        os << ", ";
      }
    }
    os << "]";
    return os;
  }
};

template <typename T, typename Alloc>
struct ObjectPrinter<std::vector<T, Alloc>> {
  template <typename Os>
  static Os &run(Os &os, const std::vector<T, Alloc> &obj) {
    os << "[";
    for (auto it = obj.cbegin(); it != obj.cend(); ++it) {
      ObjectPrinter<T>::run(os, *it);
      if (std::next(it) != obj.cend()) {
        os << ", ";
      }
    }
    os << "]";
    return os;
  }
};

template <typename T, typename Comp, typename Alloc>
struct ObjectPrinter<std::set<T, Comp, Alloc>> {
  template <typename Os>
  static Os &run(Os &os, const std::set<T, Comp, Alloc> &obj) {
    os << "{";
    for (auto it = obj.cbegin(); it != obj.cend(); ++it) {
      ObjectPrinter<T>::run(os, *it);
      if (std::next(it) != obj.cend()) {
        os << ", ";
      }
    }
    os << "}";
    return os;
  }
};

template <typename T, typename Hash, typename Pred, typename Alloc>
struct ObjectPrinter<std::unordered_set<T, Hash, Pred, Alloc>> {
  template <typename Os>
  static Os &run(Os &os, const std::unordered_set<T, Hash, Pred, Alloc> &obj) {
    os << "{";
    for (auto it = obj.cbegin(); it != obj.cend(); ++it) {
      ObjectPrinter<T>::run(os, *it);
      if (std::next(it) != obj.cend()) {
        os << ", ";
      }
    }
    os << "}";
    return os;
  }
};

template <typename K, typename V, typename Comp, typename Alloc>
struct ObjectPrinter<std::map<K, V, Comp, Alloc>> {
  template <typename Os>
  static Os &run(Os &os, const std::map<K, V, Comp, Alloc> &obj) {
    os << "{";
    for (auto it = obj.cbegin(); it != obj.cend(); ++it) {
      ObjectPrinter<K>::run(os, it->first);
      os << ":";
      ObjectPrinter<V>::run(os, it->second);
      if (std::next(it) != obj.cend()) {
        os << ", ";
      }
    }
    os << "}";
    return os;
  }
};

template <typename K, typename V, typename Hash, typename Pred, typename Alloc>
struct ObjectPrinter<std::unordered_map<K, V, Hash, Pred, Alloc>> {
  template <typename Os>
  static Os &run(Os &os,
                 const std::unordered_map<K, V, Hash, Pred, Alloc> &obj) {
    os << "{";
    for (auto it = obj.cbegin(); it != obj.cend(); ++it) {
      ObjectPrinter<K>::run(os, it->first);
      os << ":";
      ObjectPrinter<V>::run(os, it->second);
      if (std::next(it) != obj.cend()) {
        os << ", ";
      }
    }
    os << "}";
    return os;
  }
};

template <class... Args> struct ObjectPrinter<std::tuple<Args...>> {
  template <typename Os>
  static Os &run(Os &os, const std::tuple<Args...> &obj) {
    os << "(";
    mp_for_each<typename mp_integer_seq<
        std::make_index_sequence<sizeof...(Args)>>::type>([&](auto I) {
      constexpr int64_t Index = decltype(I)::value;
      using type = std::decay_t<mp_nth_t<Index, Args...>>;
      ObjectPrinter<type>::run(os, std::get<Index>(obj));
      if (Index != sizeof...(Args) - 1) {
        os << ", ";
      }
    });
    os << ")";
    return os;
  }
};

} // namespace detail

template <typename Os, typename T>
inline Os &custom_obj_print(Os &os, const T &obj) {
  return detail::ObjectPrinter<T>::run(os, obj);
}

template <typename F, class... Args>
decltype(auto) function_print(F &&f, Args &&... args) {
  decltype(auto) tuple_args = std::forward_as_tuple(args...);
  std::cout << "Args: ";
  custom_obj_print(std::cout, tuple_args);
  decltype(auto) res = tv::invoke(f, args...);
  std::cout << " Returns: ";
  custom_obj_print(std::cout, res);
  std::cout << std::endl;
  return res;
}

template <typename F> class FunctionIOPrinter {
public:
  FunctionIOPrinter(F func, std::string name, std::string file, int64_t line)
      : func_(std::move(func)), name_(name), file_(file), line_(line) {}
  template <class... Args> decltype(auto) operator()(Args &&... args) {
    std::cout << file_ << "(" << line_ << ")(" << name_ << ")";
    return function_print(func_, args...);
  }

private:
  F func_;
  std::string name_, file_;
  int64_t line_;
};

template <typename F>
FunctionIOPrinter<F> function_io_printer(F &&f, std::string name,
                                         std::string file, int64_t line) {
  return FunctionIOPrinter<F>(std::forward<F>(f), name, file, line);
}

} // namespace tv

#define TV_PRINT_IO(func)                                                      \
  tv::function_io_printer(                                                     \
      [&](auto &&... args) -> decltype(auto) {                                 \
        return func(std::forward<decltype(args)>(args)...);                    \
      },                                                                       \
      #func, __FILE__, __LINE__)
