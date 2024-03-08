#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include <vector>
#include <array>
#include <multicut/mutex-grid.hxx>
#include <multicut/mutex-grid-plus.hxx>


namespace py = pybind11;


// TODO: Make the dimension a compile time parameter?
class HyperGrid
{
public:

    HyperGrid(const std::vector<size_t>& shape, const py::array_t<int>& offsets)
    :
        n_dim_(shape.size()),
        n_offsets_(offsets.shape(0)),
        shape_(shape),
        offsets_(offsets),
        strides_(shape.size(), 1)
    {
        number_of_vertices_ = 1;
        for (size_t i = 0; i < shape_.size(); ++i)
        {
            number_of_vertices_ *= shape_[i];
            for (size_t j = 0; j < i; j++)
                strides_[j] *= shape_[i];
        }

        number_of_edges_ = number_of_vertices_ * offsets.shape(0);
    }

    size_t numberOfVertices() const
    {
        return number_of_vertices_;
    }

    size_t numberOfEdges() const
    {
        return number_of_edges_;
    }

    size_t vertexOfEdge(size_t e, size_t vertex_idx) const
    {   
        size_t source = e / n_offsets_;
        if (vertex_idx == 0)
            return source;

        size_t offset_idx = e % n_offsets_;
        e /= n_offsets_;
        
        // compute the index of target vertex
        size_t target = 0;
        for (size_t i = 0; i < n_dim_; i++){
            size_t coord = offsets_.at(offset_idx, i) + e / strides_[i];
            if ((coord < 0) || (coord >= shape_[i]))
                return source;
            target += coord * strides_[i];
            e %= strides_[i];
        }
        return target;
    }


private:
    const size_t n_dim_;
    const size_t n_offsets_;
    py::array_t<int> offsets_;
    std::vector<size_t> shape_;
    std::vector<size_t> strides_;
    size_t number_of_vertices_;
    size_t number_of_edges_;
};

template<typename Mutex, const char D>
Mutex init_grid_mutex(
    const py::array_t<size_t>& shape,
    const py::array_t<int>& connectivity,
    size_t separating_channel
) {
    // assert that the shape is of the correct shape
    if (shape.ndim() != 1 || shape.shape()[0] != D)
        throw std::runtime_error("wrong shape");
    // assert that connectivity is of the correct shape
    if (connectivity.ndim() != 2 || connectivity.shape()[1] != D)
        throw std::runtime_error("wrong connectivity");

    // read the numpy arrays as std vectors
    std::array<size_t, D> shape_array;
    std::memcpy(shape_array.data(), shape.data(), shape.size()*sizeof(size_t));

    std::vector<std::array<int, D>> connectivity_vector(connectivity.shape()[0]);
    for (size_t i = 0; i < connectivity.shape()[0]; ++i)
    {
        for (size_t j = 0; j < D; ++j)
        {
            connectivity_vector[i][j] = connectivity.at(i, j);
        }
    }
    return Mutex(shape_array, connectivity_vector, separating_channel);
}

typedef andres::graph::multicut::GridMutex<3> GridMutex3D;
typedef andres::graph::multicut::GridMutex<2> GridMutex2D;
typedef andres::graph::multicut::GridMutexPlus<3> GridMutexPlus3D;


// wrap as Python module
PYBIND11_MODULE(multi_cut, m)
{
    m.doc() = "pybind11 plugin for multi-cut heuristics";

    // TODO: Find a better solution for the grid dimension template parameter
    py::class_<GridMutex3D>(m, "GridMutex3D")
        .def(py::init(&init_grid_mutex<GridMutex3D, 3>))
        .def("run", &GridMutex3D::run<py::array_t<size_t>>)
        .def("vertex_labels", &GridMutex3D::vertex_labels);

    py::class_<GridMutex2D>(m, "GridMutex2D")
        .def(py::init(&init_grid_mutex<GridMutex2D, 2>))
        .def("run", &GridMutex2D::run<py::array_t<size_t>>)
        .def("vertex_labels", &GridMutex2D::vertex_labels);

    py::class_<GridMutexPlus3D>(m, "GridMutexPlus3D")
        .def(py::init(&init_grid_mutex<GridMutexPlus3D, 3>))
        .def("run", &GridMutexPlus3D::run<py::array_t<size_t>>)
        .def("vertex_labels", &GridMutexPlus3D::vertex_labels);
}