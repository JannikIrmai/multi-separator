#include <vector>
#include <set>
#include <array>
#include <tuple>
#include <numeric>
#include <functional>
#include <algorithm>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <andres/partition.hxx>
// TODO: Clean includes


namespace multi_separator
{

template<const char D>
class GridMutex {

public:

    /**
     * @brief Construct a new Mutex Grid object. The total number of pixels and the strides are computed.
     * 
     * @param shape - shape of the D-dimensional grid
     * @param connectivity - edge connectivity of the repulsive edges 
     */
    GridMutex(std::array<size_t, D> shape, std::vector<std::array<int, D>> connectivity)
    :
        shape(shape),
        connectivity(connectivity),
        strides(),
        partition(),
        mutexes(),
        separator()
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
        // setup the vector for the mutex graph
        mutexes.resize(n);
        // create the separator vector with all ones
        separator = std::vector<size_t>(n, 1);
        // setup the union find data structure
        partition.assign(n);
    }

    template<typename SORTED_INDICES>
    std::vector<size_t> solve(const SORTED_INDICES& sorted_indices) {
        // iterate over all indices in the sorted array
        for (size_t counter = 0; counter < sorted_indices.size(); ++counter)
        {
            // print out the progress in percent
            if (sorted_indices.size() >= 1000 && counter % (sorted_indices.size() / 1000) == 0){
                auto percent = (double)(counter * 100) / sorted_indices.size();
                std::cout << "\r" << std::fixed << std::setprecision(1) << percent << " %          " << std::flush;
            }
            auto idx = sorted_indices.at(counter);
            // check if idx refers to a pixel or to an edge
            if (idx % (connectivity.size() + 1) == 0){
                size_t u = idx / (connectivity.size() + 1);  // u is the pixel corresponding to the index
                auto coords_u = pixel_to_coordinates(u);
                // compute the roots of the grid neighbors of u
                std::set<size_t> neighbor_roots;
                for (size_t i = 0; i < D; i++){
                    for (int s : {-1, 1}) {
                        // change the i-th dimension of the coordinate of u by +- 1 to obtain 
                        // the coordinate of the neighbor v
                        std::array<int, D> coords_v;
                        for (size_t j = 0; j < D; ++j){
                            coords_v[j] = coords_u[j];
                        }
                        coords_v[i] += s;
                        auto pixel = coordinates_to_pixel(coords_v);
                        // if the coordinate of v is not valid continue
                        if (!pixel.first)
                            continue;                        
                        // if v is part of the separator continue
                        if (separator[pixel.second] == 1)
                            continue;
                        // insert the root of v into the set of neighbor roots of u
                        size_t root_v = partition.find(pixel.second);
                        // if there is a mutex constraint between u and root_v, 
                        // then u cannot be remove from the separator
                        if (check_mutex(u, root_v))
                            goto THE_END;
                        
                        neighbor_roots.insert(root_v);
                    }
                }
                // check if there is a mutex constraint between any pair of the neighboring roots
                for (auto it1 = neighbor_roots.begin(); it1 != neighbor_roots.end(); ++it1) {
                    for (auto it2 = neighbor_roots.begin(); it2 != it1; ++it2) {
                        if (check_mutex(*it1, *it2)){
                            goto THE_END;
                        }
                    }   
                }
                // mark that u is no longer in the separator
                separator[u] = 0;
                // merge all roots together with u
                for (auto root : neighbor_roots){
                    merge_roots(u, root);
                    u = partition.find(u);
                }
                
                THE_END: {}
            }
            else {
                // add a mutex constrained
                add_mutex(idx);
            }
        }
        std::cout << "        \r";  // clear the percentage print out

        return separator;
    }

    const std::vector<size_t>& get_separator() 
    {
        return separator; 
    }

    std::vector<size_t> vertex_labels(size_t ignore_comp_size = 0) 
    {
        // counter the number of elements in each component that is not part of the separator
        std::map<size_t, size_t> counts;
        for (size_t i = 0; i < n; ++i)
            if (separator[i] == 0)
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
            if (separator[i] == 0)
                labels[i] = counts[partition.find(i)];
        
        return labels;
    }


private:
    size_t n;  // total number of pixels in the grid
    std::array<size_t, D> shape;  // shape of the D dimensional grid
    std::array<size_t, D> strides;  // stride in each dimension
    std::vector<std::array<int, D>> connectivity;  // edge connectivity of the grid
    std::vector<std::set<size_t>> mutexes;  // vector that contains all mutex nodes for each root in the union find data structure
    std::vector<size_t> separator;  // characteristic vector that indicates whether a pixel is part of the separator or not
    andres::Partition<size_t> partition;

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
        size_t u = e / (connectivity.size() + 1);
        if (u >= n){
            throw std::runtime_error("Invalid edge index " + std::to_string(e) + ": to large.");
        }
        // get the connectivity offset
        int offset = e % (connectivity.size() + 1);
        if (offset == 0){
            throw std::runtime_error("Invalid edge index " + std::to_string(e) + ": is pixel.");
        }
        auto shift = connectivity[offset - 1];
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

    void add_mutex(size_t e) {
        // get the vertices of the edge
        auto vertices = vertices_of_edge(e);
        if (!std::get<0>(vertices)) {
            return;
        }
        // find the roots of the vertices of e
        size_t root_u = partition.find(std::get<1>(vertices));
        size_t root_v = partition.find(std::get<2>(vertices));
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

    /**
     * @brief This method merges the two roots u and v.
     * 
     * @param u 
     * @param v 
     */
    void merge_roots(size_t u, size_t v) {
        // merge in the union find data structure
        partition.merge(u, v);
        // get root of the merged component
        size_t root_uv = partition.find(u);  

        // update the mutex sets
        if (root_uv != u)
        {
            mutexes[root_uv].insert(mutexes[u].begin(), mutexes[u].end());
            for (auto & m : mutexes[u])
            {
                mutexes[m].erase(u);
                mutexes[m].insert(root_uv);
            }
        }
        if (root_uv != v)
        {
            mutexes[root_uv].insert(mutexes[v].begin(), mutexes[v].end());
            for (auto & m : mutexes[v])
            {
                mutexes[m].erase(v);
                mutexes[m].insert(root_uv);
            }  
        }
    }
};



} // namespace multi_separator


// TODO Joint functionality with mutex-gird.hxx for multicut

