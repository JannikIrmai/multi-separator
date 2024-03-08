#include <array>
#include <vector>
#include <cassert>


namespace andres {
namespace graph {
    

template<const char D>
class HyperGrid
{
public:

    HyperGrid(const std::array<size_t, D>& shape, const std::vector<std::array<int, D>>& offsets)
    :
        n_offsets_(offsets.size()),
        shape_(shape),
        offsets_(offsets)
    {
        n_vertices_ = 1;
        for (size_t i = 0; i < shape_.size(); ++i)
            n_vertices_ *= shape_[i];

        n_edges_ = n_vertices_ * offsets_.size();
    }

    size_t numberOfVertices() const
    {
        return n_vertices_;
    }

    size_t numberOfEdges() const
    {
        return n_edges_;
    }

    // vertex coordinates to vertex index
    size_t vertex(const std::array<size_t, D>& vertex) const
    {
        size_t index = vertex[D - 1];
        for(size_t i = D - 1; i > 0; --i) {
            index = index * shape_[i - 1] + vertex[i - 1];
        }
        return index;
    }

    // vertex index to coordinate
    std::array<size_t, D> vertex(size_t vertex) const
    {
        assert(vertex < numberOfVertices());
        std::array<size_t, D> coordinates;
        size_t i;
        for(i = 0; i < D - 1; ++i) {
            coordinates[i] = vertex % shape_[i];
            vertex = vertex / shape_[i];
        }
        coordinates[i] = vertex;
        return coordinates;
    }

    size_t vertexOfEdge(size_t e, size_t vertex_idx) const
    {   
        assert(vertex_idx < 2);
        assert(e < numberOfEdges());

        size_t offset_idx = e % n_offsets_;
        size_t source = e / n_offsets_;
        
        auto source_coords = vertex(source);
        auto target_coords = source_coords;
        for (size_t i = 0; i < D; ++i)
        {
            target_coords[i] += offsets_[offset_idx][i]; 
            if (((offsets_[offset_idx][i] < 0) && (source_coords[i] < -offsets_[offset_idx][i]) || (target_coords[i] >= shape_[i])))
                throw std::runtime_error("Invalid edge index " + std::to_string(e));
        }
        if (vertex_idx == 0)
            return source;
        else
            return vertex(target_coords);
    }


private:
    const size_t n_offsets_;
    size_t n_vertices_;
    size_t n_edges_;
    std::vector<std::array<int, D>> offsets_;
    std::array<size_t, D> shape_;
};

}
}