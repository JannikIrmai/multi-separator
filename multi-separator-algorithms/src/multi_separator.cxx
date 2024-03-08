#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include <vector>
#include <algorithm>
#include <cmath>

#include <multi-separator/mutex.hxx>
#include <multi-separator/greedy-separator-shrinking.hxx>
#include <multi-separator/mutat.hxx>
#include <multi-separator/greedy-separator-growing.hxx>

#include <andres/graph/graph.hxx>
#include <andres/graph/grid-graph.hxx>


// TODO: Be consistent in naming the offsets/connectivity

namespace py = pybind11;


typedef multi_separator::GreedySeparatorShrinking<double> GSS;

// TODO: Find a better solution for the grid dimension template parameter
typedef multi_separator::GridMutex<2> GridMutex2D;
typedef multi_separator::GridMutex<3> GridMutex3D;
typedef multi_separator::GridMutat<2> GridMutat2D;
typedef multi_separator::GridMutat<3> GridMutat3D;

typedef andres::graph::Graph<> Graph;
typedef andres::graph::GridGraph<2> GridGraph2D;
typedef andres::graph::GridGraph<3> GridGraph3D;
typedef multi_separator::GreedySeparatorGrowing<GridGraph2D, Graph, double> GSG2D;
typedef multi_separator::GreedySeparatorGrowing<GridGraph3D, Graph, double> GSG3D;


int round_halfway_to_even(double in)
{
    // round to nearest int such that halfway is rounded toward nearest even
    int out = std::lround(in);
    if (std::abs(in - out) == 0.5 && (std::abs(out) % 2 == 1))
        out += 2*(in - out);
    return out;
}


double median(std::vector<double>& v)
{
    auto n = v.size() / 2;
    std::nth_element(v.begin(), v.begin()+n, v.end());
    auto med = v[n];
    // if the length of the vector is even compute the mean of the two elements
    if(!(v.size() & 1)) 
    {
        auto max_it = max_element(v.begin(), v.begin()+n);
        med = (*max_it + med) / 2.0;
    }
    return med;    
}


template<const char D>
void py_to_std(
    const py::array_t<size_t>& shape_py,
    const py::array_t<int>& connectivity_py,
    std::array<size_t, D>& shape_array,
    std::vector<std::array<int, D>>& connectivity_vector
) {
    // assert that shape_py is of the correct shape
    if (shape_py.ndim() != 1 || shape_py.shape(0) != D)
        throw std::runtime_error("wrong shape");
    // assert that connectivity_py is of the correct shape
    if (connectivity_py.ndim() != 2 || connectivity_py.shape(1) != D)
        throw std::runtime_error("wrong connectivity");

    // copy the numpy data into the std data
    std::memcpy(shape_array.data(), shape_py.data(), shape_py.size()*sizeof(size_t));

    connectivity_vector.resize(connectivity_py.shape(0));
    for (size_t i = 0; i < connectivity_py.shape(0); ++i)
        for (size_t j = 0; j < D; ++j)
            connectivity_vector[i][j] = connectivity_py.at(i, j);
}


template<const char D>
multi_separator::GridMutex<D> init_grid_mutex(
    const py::array_t<size_t>& shape,
    const py::array_t<int>& connectivity
) {
    // read the numpy arrays as std vectors
    std::array<size_t, D> shape_array;
    std::vector<std::array<int, D>> connectivity_vector;
    py_to_std<D>(shape, connectivity, shape_array, connectivity_vector);
    // create grid mutex instance
    return multi_separator::GridMutex<D>(shape_array, connectivity_vector);
}

template<const char D>
multi_separator::GridMutat<D> init_grid_mutat(
    const py::array_t<size_t>& shape,
    const py::array_t<int>& connectivity

) {
    // read the numpy arrays as std vectors
    std::array<size_t, D> shape_array;
    std::vector<std::array<int, D>> connectivity_vector;
    py_to_std<D>(shape, connectivity, shape_array, connectivity_vector);
    // create grid mutat instance
    return multi_separator::GridMutat<D>(shape_array, connectivity_vector);
}

template<const char D>
multi_separator::GreedySeparatorGrowing<andres::graph::GridGraph<D>, Graph, double> init_grid_gsg_full(
    const py::array_t<size_t>& shape, 
    const py::array_t<int>& offsets, 
    const py::array_t<double>& costs
) {
    typedef andres::graph::GridGraph<D> GridGraph;
    typedef andres::graph::Graph<> Graph;

    std::array<size_t, D> shape_array;
    std::vector<std::array<int, D>> offsets_vector;
    py_to_std<D>(shape, offsets, shape_array, offsets_vector);
    GridGraph graph(shape_array);
    Graph interaction_graph(graph.numberOfVertices());

    std::vector<double> vertex_costs(graph.numberOfVertices());
    std::vector<double> interaction_costs;
    interaction_costs.reserve(offsets_vector.size() * graph.numberOfVertices());

    for (size_t u = 0; u < interaction_graph.numberOfVertices(); ++u)
    {   
        vertex_costs[u] = costs.at(u * (offsets_vector.size() + 1));
        std::array<size_t, D> coord_u;
        graph.vertex(u, coord_u);
        for (size_t o = 0; o < offsets_vector.size(); ++o)
        {
            bool invalid_interaction = false;
            std::array<size_t, D> coord_v(coord_u);
            for (size_t d = 0; d < D; ++d)
            {
                if (coord_v[d] + offsets_vector[o][d] < 0 || coord_v[d] + offsets_vector[o][d] >= shape_array[d])
                {
                    invalid_interaction = true;
                    break;
                }
                coord_v[d] += offsets_vector[o][d];
            }
            if (invalid_interaction)
                continue;

            size_t v = graph.vertex(coord_v);
            double c = costs.at(u * (offsets_vector.size() + 1) + o + 1);

            if (!graph.findEdge(u, v).first && c <= 0)
                continue;  // if the interaction is not an edge of the graph and if the cost is negative, continue

            interaction_graph.insertEdge(u, v);
            interaction_costs.push_back(c);
        }
    }
    return multi_separator::GreedySeparatorGrowing<GridGraph, Graph, double>(graph, interaction_graph, vertex_costs, interaction_costs);
}


template<const char D>
multi_separator::GreedySeparatorGrowing<andres::graph::GridGraph<D>, Graph, double> init_grid_gsg_line_costs_median(
    const py::array_t<size_t>& shape, 
    const py::array_t<int>& offsets,
    const py::array_t<double>& costs
) {
    typedef andres::graph::GridGraph<D> GridGraph;
    typedef andres::graph::Graph<> Graph;

    std::array<size_t, D> shape_array;
    std::vector<std::array<int, D>> offsets_vector;
    py_to_std<D>(shape, offsets, shape_array, offsets_vector);
    GridGraph graph(shape_array);
    Graph interaction_graph(graph.numberOfVertices());

    std::vector<double> vertex_costs(graph.numberOfVertices());
    for (size_t u = 0; u < interaction_graph.numberOfVertices(); ++u)
        vertex_costs[u] = costs.at(u);

    std::vector<double> interaction_costs;
    // interaction_costs.reserve(offsets_vector.size() * graph.numberOfVertices());

    std::array<size_t, D> coord_u;
    std::array<size_t, D> coord_v;
    std::array<size_t, D> coord_w;

    for (size_t o = 0; o < offsets_vector.size(); ++o)
    {
        // compute the dimension in which the offset vector goes the furthest
        size_t max_d = 0;
        for (size_t d = 1; d < D; ++d)
            if (std::abs(offsets_vector[o][d]) > std::abs(offsets_vector[o][max_d]))
                max_d = d;
        // compute the coordinates on the line defined by the offset vector by iterating over the dimension max_d
        int length = abs(offsets_vector[o][max_d]);
        std::vector<std::array<int, D>> line(length+1);
        for(size_t i = 0; i <= length; ++i)
            for (size_t d = 0; d < D; ++d)
                line[i][d] = round_halfway_to_even((double)i * offsets_vector[o][d] / length);

        std::vector<double> line_values(line.size());

        // iterate over all nodes and compute the costs of the interaction between that node and the node with the given offset.
        for (size_t u = 0; u < interaction_graph.numberOfVertices(); ++u)
        {   
            // compute the coordinates of the node u in the grid graph
            graph.vertex(u, coord_u);
            // compute the coordinate of the node whose distance to u is the offset
            bool invalid_interaction = false;
            for (size_t d = 0; d < D; ++d)
            {
                if (coord_u[d] + offsets_vector[o][d] < 0 || coord_u[d] + offsets_vector[o][d] >= shape_array[d])
                {
                    invalid_interaction = true;
                    break;
                }
                coord_v[d] = coord_u[d] + offsets_vector[o][d];
            }
            if (invalid_interaction)
                continue;
            // compute the index of the node given by the coordinates
            size_t v = graph.vertex(coord_v);
            // fill the line_values vector
            for (size_t i = 0; i < line.size(); ++i)
            {
                for (size_t d = 0; d < D; ++d)
                    coord_w[d] = coord_u[d] + line[i][d];
                size_t w = graph.vertex(coord_w);
                line_values[i] = vertex_costs[w];
            }
            // compute the median of the values on the line
            double med = median(line_values);

            if (!graph.findEdge(u, v).first && med <= 0)
                continue;  // if the interaction is not an edge of the graph and if the cost is negative, continue

            interaction_graph.insertEdge(u, v);
            interaction_costs.push_back(med);
        }
    }
    multi_separator::GreedySeparatorGrowing<GridGraph, Graph, double> gsg(graph, interaction_graph, vertex_costs, interaction_costs); 
    return gsg;
}

template<const char D>
multi_separator::GreedySeparatorGrowing<andres::graph::GridGraph<D>, Graph, double> init_grid_gsg(
    const py::array_t<size_t>& shape, 
    const py::array_t<int>& offsets,
    const py::array_t<double>& costs
) {
    size_t size = 1;
    for (auto s = shape.data(); s < shape.data() + shape.size(); ++s)
        size *= *s;
    
    if (costs.size() == size)
        return init_grid_gsg_line_costs_median<D>(shape, offsets, costs);
    else if (costs.size() == (size * (offsets.shape(0) + 1)))
        return init_grid_gsg_full<D>(shape, offsets, costs);
    else
        throw std::runtime_error("Shape of cost array does not match specified shape.");
}


// wrap as Python module
PYBIND11_MODULE(multi_separator, m)
{
    m.doc() = "pybind11 plugin for multi-separator heuristics";

    py::class_<GSS>(m, "GreedySeparatorShrinking") 
        .def(py::init<>())
        .def("setup_grid", &GSS::setup_grid<py::array_t<size_t>, py::array_t<int>, py::array_t<double>, py::array_t<double>>)
        .def("run", &GSS::run, "Run the greedy separator shrinking algorithm", py::arg("max_iter") = -1)
        .def("num_iter", &GSS::num_iter)
        .def("objective", &GSS::objective)
        .def("separator", &GSS::separator)
        .def("vertex_labels", &GSS::vertex_labels);

    py::class_<GSG2D>(m, "GreedySeparatorGrowing2D")
        .def(py::init(&init_grid_gsg<2>))
        .def("run", &GSG2D::run, py::arg("max_iter") = size_t(-1))
        .def("vertex_labels", &GSG2D::vertex_labels)
        .def("remove_small_components", &GSG2D::remove_small_components)
        .def("num_iter", &GSG2D::num_iter);

    py::class_<GSG3D>(m, "GreedySeparatorGrowing3D")
        .def(py::init(&init_grid_gsg<3>))
        .def("run", &GSG3D::run, py::arg("max_iter") = size_t(-1))
        .def("vertex_labels", &GSG3D::vertex_labels)
        .def("remove_small_components", &GSG3D::remove_small_components)
        .def("recompute_potentials", &GSG3D::recompute_potentials)
        .def("get_potential", &GSG3D::get_potential)
        .def("num_iter", &GSG3D::num_iter); 

    py::class_<GridMutex3D>(m, "GridMutex3D")
        .def(py::init(&init_grid_mutex<3>))
        .def("run", &GridMutex3D::solve<py::array_t<size_t>>)
        .def("separator", &GridMutex3D::get_separator)
        .def("vertex_labels", &GridMutex3D::vertex_labels);

    py::class_<GridMutex2D>(m, "GridMutex2D")
        .def(py::init(&init_grid_mutex<2>))
        .def("run", &GridMutex2D::solve<py::array_t<size_t>>)
        .def("separator", &GridMutex2D::get_separator)
        .def("vertex_labels", &GridMutex2D::vertex_labels);

    py::class_<GridMutat2D>(m, "GridMutat2D")   
        .def(py::init(&init_grid_mutat<2>))
        .def("run", &GridMutat2D::run_container<py::array_t<size_t>>)
        .def("separator", &GridMutat2D::separator)
        .def("vertex_labels", &GridMutat2D::vertex_labels)
        .def("make_labels_consecutive", &GridMutat2D::make_labels_consecutive);  

    py::class_<GridMutat3D>(m, "GridMutat3D")   
        .def(py::init(&init_grid_mutat<3>))
        .def("run", &GridMutat3D::run_container<py::array_t<size_t>>)
        .def("separator", &GridMutat3D::separator)
        .def("vertex_labels", &GridMutat3D::vertex_labels)
        .def("make_labels_consecutive", &GridMutat3D::make_labels_consecutive);   
}