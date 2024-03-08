#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include <vector>
#include <random>

#include <synth_growth.hxx>
#include <andres/graph/grid-graph.hxx>


namespace py = pybind11;

template<const char D>
std::vector<size_t> synth_grow_python(size_t n, size_t num, size_t seed, bool separator_line)
{
    andres::graph::GridGraph<D> graph({n, n, n});

    // define vertex priorities
    std::vector<unsigned short> vertexPriorities(graph.numberOfVertices());
    {
        std::default_random_engine randomEngine;
        randomEngine.seed(seed);
        std::uniform_int_distribution<unsigned short> priorityDistribution(0, 4096);
        for(auto & v : vertexPriorities) {
            v = priorityDistribution(randomEngine);
        }
    }

    std::vector<size_t> componentLabeling(graph.numberOfVertices());
    growDecomposition(
        graph, vertexPriorities.begin(), vertexPriorities.end(), 
        num, componentLabeling.begin(), componentLabeling.end(), 
        seed, separator_line);
    return componentLabeling;
}



// wrap as Python module
PYBIND11_MODULE(synth_growth, m)
{
    m.doc() = "pybind11 plugin generating synthetic cell like segmentation masks";

    m.def("synth_growth_2d", &synth_grow_python<2>, py::arg("n"), py::arg("num"), py::arg("seed"), py::arg("separator_line") = true);
    m.def("synth_growth_3d", &synth_grow_python<3>, py::arg("n"), py::arg("num"), py::arg("seed"), py::arg("separator_line") = true);
}