#pragma once 

namespace tv {

namespace cuda {
namespace Arch {
enum _Arch {
    kSm50 = 50,
    kSm60 = 60, // Titan X Pascal
    kSm61 = 61, // GTX 1000
    kSm70 = 70, // Titan V, Tesla V100
    kSm72 = 72, // Xavier
    kSm75 = 75, // RTX 2000
    kSm80 = 80, // A100
    kSm86 = 86, // RTX 3000
    kEnd = 999999
};
}
}

}