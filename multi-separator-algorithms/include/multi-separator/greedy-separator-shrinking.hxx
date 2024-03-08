#pragma once
#ifndef MULTI_SEPARATOR_GREEDY_SEPARATOR_SHRINKING_HXX
#define MULTI_SEPARATOR_GREEDY_SEPARATOR_SHRINKING_HXX

#include <cstddef>
#include <iterator>
#include <vector>
#include <algorithm>
#include <map>
#include <andres/random-access-set.hxx>
#include <queue>
#include "andres/partition.hxx"
#include <iostream>

namespace multi_separator {

// This is a simple implementation of an undirected graph which can be efficiently manipulated.
class DynamicGraph
{
public:
    DynamicGraph(size_t n) :
        vertices_(n)
    {}

    size_t size()
    {
        return vertices_.size();
    }

    bool edgeExists(size_t a, size_t b) const
    {
        if (vertices_[a].empty())
            return false;
        else if (vertices_[a].size() < vertices_[b].size())
            return vertices_[a].count(b);
        else 
            return vertices_[b].count(a);
    }

    andres::RandomAccessSet<size_t> const& getAdjacentVertices(size_t v) const
    {
        return vertices_[v];
    }

    void removeVertex(size_t v)
    {
        for (auto& n : vertices_[v])
            vertices_[n].erase(v);

        vertices_[v] = andres::RandomAccessSet<size_t>();
    }

    void insertEdge(size_t a, size_t b)
    {
        vertices_[a].insert(b);
        vertices_[b].insert(a);
    }

private:
    std::vector<andres::RandomAccessSet<size_t>> vertices_;
};

// This is a simple implementation of an undirected edge-weighted graph that can be efficiently manipulated.
template<typename T>
class DynamicWeightedGraph
{
public:

    struct WeightedVertex
    {
        WeightedVertex(size_t v, T w): v_(v), w_(w) {}
    
        bool operator<(const WeightedVertex& rhs) const
        {
            return v_ < rhs.v_;
        }
        size_t v_;
        T w_;
    };

    DynamicWeightedGraph(size_t n) :
        vertices_(n),
        vertex_weights_(n)
    {}

    size_t size()
    {
        return vertices_.size();
    }

    bool edgeExists(size_t a, size_t b) const
    {
        if (vertices_[a].empty())
            return false;
        else if (vertices_[a].size() < vertices_[b].size())
        {
            WeightedVertex vertex_b(b, 0);
            return vertices_[a].count(vertex_b);
        }
        else 
        {
            WeightedVertex vertex_a(a, 0);
            return vertices_[b].count(vertex_a);
        }
    }

    andres::RandomAccessSet<WeightedVertex> const& getAdjacentVertices(size_t v) const
    {
        return vertices_[v];
    }

    T getEdgeWeight(size_t a, size_t b) const
    {
        if (vertices_[a].size() < vertices_[b].size())
        {
            WeightedVertex vertex_b(b, 0);
            return vertices_[a].find(vertex_b)->w_;
        }
        else
        {
            WeightedVertex vertex_a(a, 0);
            return vertices_[b].find(vertex_a)->w_;
        }
    }

    void removeVertex(size_t v)
    {
        WeightedVertex vertex_v(v, 0);
        for (auto& n: vertices_[v])
            vertices_[n.v_].erase(vertex_v);
        vertices_[v] = andres::RandomAccessSet<WeightedVertex>();
    }

    void updateEdgeWeight(size_t a, size_t b, T w)
    {
        WeightedVertex vertex_a(a, w);
        WeightedVertex vertex_b(b, w);

        auto it = vertices_[a].find(vertex_b);
        if (it != vertices_[a].end())
        {
            it->w_ += w;
            it = vertices_[b].find(vertex_a);
            it->w_ += w;
        }
        else 
        {
            vertices_[a].insert(vertex_b);
            vertices_[b].insert(vertex_a);
        }
    }

    void setVertexWeight(size_t v, T w) 
    {
        vertex_weights_[v] = w;
    }

    T getVertexWeight(size_t v)
    {
        return vertex_weights_[v];
    }

private:
    std::vector<andres::RandomAccessSet<WeightedVertex>> vertices_;
    std::vector<T> vertex_weights_;
};

template<typename T>
class GreedySeparatorShrinking
{
public:

    struct Vertex
    {
        Vertex(size_t _v, T _p)
        {
            v = _v;
            p = _p;
            edition = 0;
        }

        size_t v;  // vertex index
        size_t edition;
        T p;  // vertex potential

        bool operator <(Vertex const& other) const
        {
            return p < other.p;
        }
    };

    GreedySeparatorShrinking() : graph_(0), interaction_graph_(0) {}

    GreedySeparatorShrinking(const DynamicGraph& graph, const DynamicWeightedGraph<T>& interaction_graph):
        graph_(graph), interaction_graph_(interaction_graph)
    {
        setup_();
    }

    template<typename GRAPH, typename INTERACTION_GRAPH, typename VERTEX_COSTS, typename INTERACTION_COSTS>
    GreedySeparatorShrinking(
        const GRAPH& graph,
        const INTERACTION_GRAPH& interaction_graph,
        const VERTEX_COSTS& vertex_costs,
        const INTERACTION_COSTS& interaction_costs
    ):
        graph_(graph.numberOfVertices()), interaction_graph_(interaction_graph.numberOfVertices())
    {   
        // construct a dynamic copy of the base graph
        for (size_t i = 0; i < graph.numberOfEdges(); ++i) 
        {
            auto a = graph.vertexOfEdge(i, 0);
            auto b = graph.vertexOfEdge(i, 1);
            graph_.insertEdge(a, b);
        }
        // construct a dynamic copy of the interaction graph
        for (size_t i = 0; i < interaction_graph.numberOfVertices(); ++i)
        {
            interaction_graph_.setVertexWeight(i, vertex_costs[i]);
        }
        for (size_t i = 0; i < interaction_graph.numberOfEdges(); ++i) 
        {
            auto a = interaction_graph.vertexOfEdge(i, 0);
            auto b = interaction_graph.vertexOfEdge(i, 1);
            interaction_graph_.updateEdgeWeight(a, b, interaction_costs[i]);
        }
        setup_();
    }

    GreedySeparatorShrinking(
        const std::vector<size_t>& shape, 
        const std::vector<int>& connectivity,
        const std::vector<T>& vertex_costs, 
        const std::vector<T>& interaction_costs
    ) :
        GreedySeparatorShrinking() 
    {
        setup_grid(shape, connectivity, vertex_costs, interaction_costs);
    }
    
    void run(int max_iter = -1)
    {
        size_t num_iter_start = num_iter_;
        while (!queue_.empty())
        {
            auto vertex = queue_.top();
            queue_.pop();
            // continue if the vertex edition is not the most recent one
            if (vertex.edition < vertex_editions_[vertex.v])
                continue;
            // continue of the potential is negative
            if (vertex.p < T())
                break;

            // break the loop is the maximum number of iterations is reached
            if (max_iter >= 0 && num_iter_ >= num_iter_start + max_iter)
                break;
            ++num_iter_;

            // printout the progress as a percentage
            if (graph_.size() >= 100 && num_iter_ % (graph_.size() / 100) == 0){
                auto percent = (num_iter_ * 100) / graph_.size();
                std::cout << "\r" << percent << " %" << std::flush;
            }

            obj_ -= vertex.p;
            // remove the vertex from the separator
            remove_vertex_(vertex.v);
        }
        std::cout << "\r";  // clear the percentage print out
    }

    const auto& separator() const
    {
        return separator_;
    }

    T objective() const
    {
        return obj_;
    }

    size_t num_iter() const
    {
        return num_iter_;
    }

    /**
     * @brief This method returns a vector that labels all vertices such that vertices that
     *  are in the separator have label 0, and all other vertices have labels != 0 such that
     *  two vertices that are in the same connected component with respect to the separator
     *  have the same label. The labels are consecutive, i.e. each label from 0 to
     *  the number of connected components exists.
     * 
     * @return std::vector<size_t> 
     */
    std::vector<size_t> vertex_labels()
    {
        std::map<size_t, size_t> indices;
        for (size_t v = 0; v < graph_.size(); ++v)
            indices[partition_.find(v)] = 0;

        size_t cnt = 1;
        for (auto& p : indices)
            if (separator_[p.first] == 0)
                p.second = cnt++;

        std::vector<size_t> labels(graph_.size(), 0);
        for (size_t v = 0; v < graph_.size(); ++v)
            if (separator_[v] == 0)
                labels[v] = indices[partition_.find(v)];
        
        return labels;
    }

    template<typename SHAPE, typename CONNECTIVITY, typename VERTEX_COSTS, typename INTERACTION_COSTS>
    void setup_grid(
        const SHAPE& shape, 
        const CONNECTIVITY& connectivity,
        const VERTEX_COSTS& vertex_costs, 
        const INTERACTION_COSTS& interaction_costs
    ) {
        const size_t D = shape.size();
        if (D == 0) {
            return;
        }
        // assert that the connectivity matches the shape
        if (connectivity.size() % D != 0)
            throw std::runtime_error("connectivity does not agree with shape.");

        // compute the total number of vertices and the strides
        size_t n = 1;
        std::vector<size_t> strides(D);
        for (size_t d = 0; d < D; ++d)
        {
            strides[d] = 1;
            n *= shape.at(d);
            for (size_t j = 0; j < d; j++)
            {
                strides[j] *= shape.at(d);
            }
        }

        // assert that the size of the vertex and interaction costs is correct
        if (vertex_costs.size() != n) 
        {
            throw std::runtime_error("size of vertex_costs does not agree with shape.");
        }
        if (interaction_costs.size() != n * (connectivity.size() / D)) 
        {
            throw std::runtime_error("size of interaction_costs does not agree with shape and size of connectivity.");
        }

        // helper function that computes the grid coordinates from a given vertex index
        auto idx_to_coordinates = [&strides, &D] (size_t idx) -> std::vector<size_t>
        {
            std::vector<size_t> coords(D);
            for (size_t d = 0; d < D; d++)
            {
                coords[d] = idx / strides[d];
                idx %= strides[d];
            }
            return coords;
        };

        // helper function that computes the index of a vertex from given coordinates
        auto coordinates_to_pixel = [&strides, &shape, &D] (std::vector<int> coords) -> std::pair<bool, size_t>
        {
            size_t pixel = 0;
            for(size_t d = 0; d < D; d++)
            {
                if (coords[d] < 0 || coords[d] >= shape.at(d))
                {
                    return {false, 0};
                }
                pixel += coords[d] * strides[d];
            }
            return {true, pixel};
        };

        // create the base grid graph
        graph_ = DynamicGraph(n);
        for (size_t i = 0; i < n; ++i)
        {
            auto coords = idx_to_coordinates(i);
            for (size_t d = 0; d < D; ++d)
            {
                if (coords[d] + 1 < shape.at(d)) {
                    graph_.insertEdge(i, i + strides[d]);
                }
            }
        }

        // create the interaction graph
        interaction_graph_ = DynamicWeightedGraph<T>(n);
        // set the vertex costs
        for (size_t i = 0; i < n; ++i) 
        {
            interaction_graph_.setVertexWeight(i, vertex_costs.at(i));
        }
        // set the interaction costs
        for (size_t i = 0; i < n; ++i) 
        {
            auto coords = idx_to_coordinates(i);
            for (size_t o = 0; o < connectivity.size() / D; ++o) 
            {
                std::vector<int> coords_neighbor(D);
                for (size_t d = 0; d < D; ++d)
                {
                    coords_neighbor[d] =  connectivity.at(o * D + d) + coords[d];
                }
                auto pix = coordinates_to_pixel(coords_neighbor);
                if (pix.first)
                    interaction_graph_.updateEdgeWeight(i, pix.second, interaction_costs.at(o * n + i));
            }
        }
        setup_();
    }

    

private:
    DynamicGraph graph_;
    DynamicWeightedGraph<T> interaction_graph_;
    std::vector<size_t> separator_;
    std::vector<T> potentials_;
    std::priority_queue<Vertex> queue_;
    std::vector<size_t> vertex_editions_;
    andres::Partition<size_t> partition_;
    size_t num_iter_;
    T obj_;

    void setup_()
    {
        if (graph_.size() != interaction_graph_.size())
        {
            throw std::runtime_error("graph and interaction_graph do not have same number of vertices.");
        }
        num_iter_ = 0;
        // compute the objective value of the solution with all vertices in the separator
        obj_ = 0.0;
        for (size_t i = 0; i < interaction_graph_.size(); ++i) 
        {
            obj_ += interaction_graph_.getVertexWeight(i);
            for (auto& n : interaction_graph_.getAdjacentVertices(i) )
            {
                obj_ += n.w_ / 2;
            }
        }

        separator_ = std::vector<size_t>(graph_.size(), 1);
        potentials_ = std::vector<T>(graph_.size());
        for (size_t i = 0; i < interaction_graph_.size(); ++i)
        {
            potentials_[i] = interaction_graph_.getVertexWeight(i);
        }
        // build the priority queue
        queue_ = std::priority_queue<Vertex>(); // TODO: Initialize queue from container for linear time construction
        vertex_editions_ = std::vector<size_t>(graph_.size(), 0);
        for (size_t i = 0; i < graph_.size(); ++i)
        {   
            // only add the vertex to the queue if its potential is non-negative
            if(potentials_[i] >= T())
            {
                auto vertex = Vertex(i, potentials_[i]);
                queue_.push(vertex);
            }
        }

        // construct the initial partition of the vertices, i.e. each vertex in its own set.
        partition_ = andres::Partition<size_t>(graph_.size());

        num_iter_ = 0;
    }

    void update_vertex_in_queue_(size_t v)
    {
        ++vertex_editions_[v];
        // only add the vertex to the queue if the potential is non negative
        if (potentials_[v] >= T())
        {
            Vertex vertex_to_update(v, potentials_[v]);
            vertex_to_update.edition = vertex_editions_[v];
            queue_.push(vertex_to_update);
        }
    }

    void remove_vertex_(size_t s)
    {
        separator_[s] = 0;

        // get all the neighboring vertices that are not part of the separator which will form the new component
        std::vector<size_t> new_comp;
        for (auto& n : graph_.getAdjacentVertices(s)){
            if (separator_[n] == 0){
                new_comp.push_back(n);
            }
        }
        // include the vertex s in the list of neighbors
        new_comp.push_back(s);

        // -------------- Update the potentials of the neighboring nodes ---------------------
        bool update_after_contraction = false;
        if (new_comp.size() == 1)
        {
            // If new_comp consists of just the current vertex s, increase the potential of all its neighbors by the cost of the interaction.
            // Further, check if there are neighbors that are now a separating vertex between s and another 
            // existing component. If so, increase the potential of those neighbors accordingly.

            for (auto& n : graph_.getAdjacentVertices(s))
            {    
                if (interaction_graph_.edgeExists(s, n))
                {
                    potentials_[n] += interaction_graph_.getEdgeWeight(s, n);
                }
                for (auto& nn : graph_.getAdjacentVertices(n))
                {
                    if (nn == s)
                        continue;
                        
                    if (separator_[nn] == 1)
                        continue;
                    
                    if (interaction_graph_.edgeExists(s, nn))
                    {
                        potentials_[n] += interaction_graph_.getEdgeWeight(s, nn);
                    }
                }
                update_vertex_in_queue_(n);
            }
        }
        else if (new_comp.size() == 2) 
        {
            // if the new component only consists of two vertices, i.e. vertex s and one other vertex, old_comp, then updating is a bit easier:
            // We need to consider the following cases:
            //  - s has a neighbor in the interaction graph that is either a neighbor of s or of old_comp in the base graph (i.e. the neighbor is part of the separator):
            //      -> (a) Increase the potential of the neighbor by the cost of the interaction from the neighbor to s.
            //  - s has a neighbor that is not a neighbor of old_comp and there is an interaction from old_comp to that neighbor:
            //      -> (b) Increase the potential of the neighbor by the cost of the interaction from old_comp to the neighbor.
            //  - s has a neighbor n in the interaction graph that is not part of the separator and n is a neighbor of a neighbor nn of old_comp:
            //      -> (c) Increase the potential of nn by the coster of the interaction from s to n.
            //  - s has a neighbor n that has a neighbor nn that is not in the separator and that is connected to old_comp or s by an interaction:
            //      -> (d) Increase the potential of n by the cost of the interaction from old_comp to nn.

            const size_t old_comp = new_comp[0];  // the already existing component where s is being merged into
            // iterate over all neighbors of s in the interaction graph.
            // NOTE: the number of neighbors is bounded by the maximum vertex degree of the input graph. In praxis this degree is (usually) a small constant.
            for (auto& n : interaction_graph_.getAdjacentVertices(s))
            {
                if (n.v_ == old_comp)
                    continue;
                
                // if the neighbor is not part of the separator that means that the neighbor is separated from old_comp and vertex s. 
                // For all vertices nn that are adjacent to both the neighbor and vertex s increase the potential by the value of the interaction.
                if (separator_[n.v_] == 0){
                    // THIS LOOP IS PROBLEMATIC: The size neighborhood of neighbor might be linear in the input size. 
                    auto it_1 = graph_.getAdjacentVertices(n.v_).begin();
                    auto it_2 = graph_.getAdjacentVertices(old_comp).begin();
                    while (it_1 != graph_.getAdjacentVertices(n.v_).end() && it_2 != graph_.getAdjacentVertices(old_comp).end()){
                        if (*it_1 < *it_2){
                            ++it_1;
                        }
                        else if (*it_1 > *it_2){
                            ++it_2;
                        }
                        else {
                            // check if *it_1 is connected to s. If not, increase the potential of *it_1 by the cost of the interaction.
                            if (!graph_.edgeExists(*it_1, s)){
                                potentials_[*it_1] += n.w_;
                                update_vertex_in_queue_(*it_1);  // (c)
                            }
                            ++it_1;
                        }
                    }
                    continue;
                }

                // if the neighbor is connected to s by an edge increase the potential by the interaction cost
                if (graph_.edgeExists(s, n.v_) || graph_.edgeExists(old_comp, n.v_))
                {
                    potentials_[n.v_] += n.w_;
                    update_vertex_in_queue_(n.v_);  // (a)
                }
            }
            // iterate over all neighbors of s in the graph and check if it is not a neighbor of old_comp. If so, check if there is an interaction
            // to old_comp, and if so, increase the potential of the neighbor by the cost of the interaction
            for (auto& n : graph_.getAdjacentVertices(s))
            {
                if (n == old_comp)
                    continue;

                if (graph_.edgeExists(old_comp, n))
                    continue;

                if (interaction_graph_.edgeExists(n, old_comp)){
                    potentials_[n] += interaction_graph_.getEdgeWeight(n, old_comp);  // (b)
                }

                // iterate over all neighbors nn of n. If nn is not in the separator, increase the potential of n by the costs
                // of the interactions from nn to s. Further, check if nn is connected to old_comp. If not, increase the potential 
                // of neighbor by the cost of the interaction from old_comp to nn.
                for (auto& nn : graph_.getAdjacentVertices(n)){
                    if (nn == s)
                        continue;
                    
                    if (nn == old_comp)
                        continue;
                    
                    if (separator_[nn] == 1)
                        continue;
                    
                    // if there is an interaction from nn to old_comp, increase the potential of neighbor by the cost of that interaction
                    if (interaction_graph_.edgeExists(old_comp, nn))
                    {
                        potentials_[n] += interaction_graph_.getEdgeWeight(old_comp, nn);  // (d)
                    }
                    // if there is an interaction from nn to s, increase the potential of neighbor by the cost of that interaction
                    if (interaction_graph_.edgeExists(s, nn))
                    {
                        potentials_[n] += interaction_graph_.getEdgeWeight(s, nn);  // (d)
                    }
                }
                // reinsert the neighbor n with the updated potential into the queue
                update_vertex_in_queue_(n);
            }
        }
        else 
        {
            update_after_contraction = true;
        }

        // ---------------- contract all vertices in new_comp into the first vertex
        while (new_comp.size() > 1)
        {
            // merge in partition
            size_t w = new_comp[new_comp.size() - 1];
            new_comp.pop_back();
            partition_.merge(new_comp[0], w);

            // contract in base graph
            for (auto& n : graph_.getAdjacentVertices(w))
            {
                if (n == new_comp[0])
                    continue;

                graph_.insertEdge(new_comp[0], n);
            }
            graph_.removeVertex(w);

            // contract in interaction graph
            for (auto& n : interaction_graph_.getAdjacentVertices(w)) 
            {
                // if the neighbor is in new_comp, then the interaction needs not to be considered
                if (std::find(new_comp.begin(), new_comp.end(), n.v_) != new_comp.end())
                    continue;

                interaction_graph_.updateEdgeWeight(new_comp[0], n.v_, n.w_);
            }
            interaction_graph_.removeVertex(w);
        }

        if (update_after_contraction)
        {
            // recompute the potential of all neighbors of new_comp
            for (auto& n : graph_.getAdjacentVertices(new_comp[0])) 
            {
                auto potential = interaction_graph_.getVertexWeight(n);
                // get the component that would be created if n was removed from the separator
                std::vector<size_t> potential_new_comp = {n};
                for (auto& nn : graph_.getAdjacentVertices(n)) 
                {
                    if (separator_[nn] == 0) {
                        potential_new_comp.push_back(nn);
                    }
                }
                // sum the values of all interactions between vertices in the potential new component
                for (size_t i = 0; i < potential_new_comp.size(); ++i)
                {
                    for (size_t j = i+1; j < potential_new_comp.size(); ++j)
                    {
                        if (interaction_graph_.edgeExists(potential_new_comp[i], potential_new_comp[j]))
                        {
                            potential += interaction_graph_.getEdgeWeight(potential_new_comp[i], potential_new_comp[j]);
                        }   
                    }
                }
                potentials_[n] = potential;
                update_vertex_in_queue_(n);
            }
        }
    }
};

} // namespace multi_separator

#endif // #ifndef MULTI_SEPARATOR_GREEDY_SEPARATOR_SHRINKING_HXX
