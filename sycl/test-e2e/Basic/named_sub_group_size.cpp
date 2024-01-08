// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

#include <sycl/sycl.hpp>

using namespace sycl;

int main() {
  queue q;

  auto dev = q.get_device();
  // auto primary_sg_size = dev.get_info<info::device::primary_sub_group_size>();

  {
    using namespace sycl::ext::oneapi;
    using namespace sycl::ext::oneapi::experimental;
    
    nd_range<1> ndr{1, 1};
    auto *out = malloc_shared<size_t>(1, q);
    q.parallel_for(ndr, properties{sub_group_size_primary}, [=](auto it) {
      *out = it.get_sub_group().get_max_local_range()[0];
    }).wait();
    // assert(*out == primary_sg_size);
    free(out, q);
    q.parallel_for(ndr, properties{sub_group_size_automatic}, [=](auto it) {});
  }
}