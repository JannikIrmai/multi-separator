#include <vector>
#include <set>
#include <array>
#include <tuple>
#include <iostream>
#include <iomanip>
#include "andres/partition.hxx"


namespace andres {
namespace graph {
namespace multicut {

template<const char D>
class GridMutex {

public:

    /**
     * @brief Construct a new Mutex Grid object. The total number of pixels and the strides are computed.
     * 
     * @param shape - shape of the D-dimensional grid
     * @param connectivity - edge connectivity of the repulsive edges 
     */
    GridMutex(std::array<size_t, D> shape, std::vector<std::array<int, D>> connectivity, size_t separating_channel)
    :
        shape(shape),
        connectivity(connectivity),
        separating_channel(separating_channel),
        strides(),
        mutexes(),
        partition()
    {
        // compute the total number of pixels in the image and the strides 
        n = 1;
        for (size_t i = 0; i < D; i++){
            strides[i] = 1;
            n *= shape[i];
            for (size_t j = 0; j < i; j++){
                strides[j] *= shape[i];
            }
        }
        // setup the union find data structure
        partition.assign(n);
        // setup the vector for the mutexes
        mutexes.resize(n);
    }

    template<typename SORTED_INDICES>
    void run(const SORTED_INDICES& sorted_indices) 
    {    
        for (size_t counter = 0; counter < sorted_indices.size(); ++counter)
        {
            // print out the progress in percent
            if (sorted_indices.size() >= 1000 && counter % (sorted_indices.size() / 1000) == 0){
                auto percent = (double)(counter * 100) / sorted_indices.size();
                std::cout << "\r" << std::fixed << std::setprecision(1) << percent << " %          " << std::flush;
            }

            size_t idx = sorted_indices.at(counter);
            
            auto e = vertices_of_edge(idx);
            // continue if the edge is not valid
            if (!std::get<0>(e))
                continue;
            // merge the components or add a mutex constraint
            if (idx % connectivity.size() < separating_channel)
                merge(std::get<1>(e), std::get<2>(e));
            else
                add_mutex(std::get<1>(e), std::get<2>(e));
        }
        std::cout << "         \r";  // clear the percentage print out
    }

    std::vector<size_t> vertex_labels(size_t ignore_comp_size = 0) 
    {
        // counter the number of elements in each component that is not part of the separator
        std::map<size_t, size_t> counts;
        for (size_t i = 0; i < n; ++i)
            ++counts[partition.find(i)];

        // map the indices of the components that are not part of the separator to consecutive integers starting from 1
        size_t idx = 1;
        for (auto& p : counts)
        {
            if (p.second <= ignore_comp_size)
                p.second = 0;
            else
                p.second = idx++;
        }

        // create the label vector
        std::vector<size_t> labels(n, 0);
        for (size_t i = 0; i < n; ++i)
            labels[i] = counts[partition.find(i)];

        return labels;
    }


private:
    size_t n;  // total number of pixels in the grid
    std::array<size_t, D> shape;  // shape of the D dimensional grid
    std::array<size_t, D> strides;  // stride in each dimension
    std::vector<std::array<int, D>> connectivity;  // edge connectivity of the grid
    std::vector<std::set<size_t>> mutexes;  // vector that contains all mutex nodes for each root in the union find data structure
    andres::Partition<size_t> partition;
    size_t separating_channel;

    /**
     * @brief This method computes the D-dimensional coordinates of a pixel index p
     * 
     * @param p 
     * @return std::array<size_t, D> 
     */
    std::array<size_t, D> pixel_to_coordinates(size_t p) const {
        std::array<size_t, D> coords;
        for (size_t i = 0; i < D; i++){
            coords[i] = p / strides[i];
            p %= strides[i];
        }
        return coords;
    }

    /**
     * @brief This method computes the index of a pixel that is specified by D-dimensional coordinates.
     *      A pair is returned where the first element indicates if the coordinates are valid and the second element 
     *      contains the index of the pixel if the coordinates are valid.
     *      
     * @return std::pair<bool, size_t> 
     */
    std::pair<bool, size_t> coordinates_to_pixel(std::array<int, D>coordinates) const {
        size_t pixel = 0;
        for(size_t i = 0; i < D; i++){
            if (coordinates[i] < 0 || coordinates[i] >= shape[i]){
                return {false, 0};
            }
            pixel += coordinates[i] * strides[i];
        }
        return {true, pixel};
    }

    /**
     * @brief This method computes the indices of the pixels that are connected by the specified edge
     * 
     * @param e 
     * @return std::tuple<bool, size_t, size_t> 
     */
    std::tuple<bool, size_t, size_t> vertices_of_edge(size_t & e) const {
        // get the outgoing index of the edge
        size_t u = e / connectivity.size();
        if (u >= n){
            throw std::runtime_error("Invalid edge index " + std::to_string(e) + ": to large.");
        }
        // get the connectivity offset
        int offset = e % connectivity.size();
        auto shift = connectivity[offset];
        auto coords_u = pixel_to_coordinates(u);
        std::array<int, D> coords_v;
        for (size_t i = 0; i < D; i++){
            coords_v[i] = shift[i] + coords_u[i];
        }
        auto pixel = coordinates_to_pixel(coords_v);
        if (pixel.first){
            return {true, u, pixel.second};
        }
        else {
            return {false, 0, 0};
        }
    }

    bool check_mutex(size_t u, size_t v) const {
        // search in smaller mutex set for the other root
        if (mutexes[u].size() < mutexes[v].size())
            return mutexes[u].find(v) != mutexes[u].end();
        else
            return mutexes[v].find(u) != mutexes[v].end();
    }

    void add_mutex(size_t u, size_t v) 
    {    
        // find the roots of the vertices
        size_t root_u = partition.find(u);
        size_t root_v = partition.find(v);
        // if root_u and root_v are equal, then u and v are already in the same component
        // and the mutex constrained cannot be added
        if (root_u == root_v){
            return;
        }
        // if there is already a mutex constrained the mutex constrained cannot be added
        if (check_mutex(root_u, root_v)) {
            return;
        }
        // add mutex constrained by inserting the edge at the right position such that the arrays remain sorted
        mutexes[root_u].insert(root_v);
        mutexes[root_v].insert(root_u);
    }

    void merge(size_t u, size_t v) 
    {
        size_t root_u = partition.find(u);
        size_t root_v = partition.find(v);
        // if there is a mutex contraint between the roots, they cannot be merged
        if (check_mutex(root_u, root_v))
            return;
        // merge in the union find data structure
        partition.merge(root_u, root_v);
        // get root of the merged component
        size_t root_uv = partition.find(root_u);  

        // update the mutex sets
        if (root_uv != root_u)
        {
            mutexes[root_uv].insert(mutexes[root_u].begin(), mutexes[root_u].end());
            for (auto & m : mutexes[root_u])
            {
                mutexes[m].erase(root_u);
                mutexes[m].insert(root_uv);
            }
        }
        if (root_uv != root_v)
        {
            mutexes[root_uv].insert(mutexes[root_v].begin(), mutexes[root_v].end());
            for (auto & m : mutexes[root_v])
            {
                mutexes[m].erase(root_v);
                mutexes[m].insert(root_uv);
            }  
        }
    }
};


} // namespace multicut
} // namespace graph
} // namespace andres
